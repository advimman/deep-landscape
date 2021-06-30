#!/usr/bin/env python3
import collections
import glob
import os

import pandas as pd
import numpy as np
import torch.nn.functional as F
import PIL.Image as Image

from inference.base_image_utils import get_scale_size, image2batch, choose_center_full_size_crop_params
from inference.metrics.fid.fid_score import _compute_statistics_of_images, \
    calculate_frechet_distance
from inference.metrics.fid.inception import InceptionV3
from inference.metrics.lpips import LPIPSLossWrapper
from inference.perspective import load_video_frames_from_folder, FlowPredictor
from inference.segmentation import SegmentationModule
from inference.encode_and_animate import calc_segmentation_posterior_error, sum_dicts
from inference.metrics.ssim import SSIM
import constants


MOVABLE_CLASSES = [2, 21]


def calc_optical_flow_metrics(flow_predictor, frames, movable_mask):
    if not movable_mask.any():
        return dict(flow_l2=float('nan'))

    assert not (frames < 0).any() and not (frames > 1).any()
    flows = flow_predictor.predict_flow(frames * 2 - 1)[1]
    flows_x, flows_y = flows[:, [0]], flows[:, [1]]
    flow_x_median = float(flows_x[movable_mask.expand_as(flows_x)].abs().mean())
    flow_y_median = float(flows_y[movable_mask.expand_as(flows_y)].abs().mean())

    result = dict(flow_l2=(flow_x_median ** 2 + flow_y_median ** 2) ** 0.5)

    return result


def batch2pil(batch):
    np_batch = ((batch.permute(0, 2, 3, 1) / 2 + 0.5) * 255).clamp(0, 255).cpu().numpy().astype('uint8')
    return [Image.fromarray(ar) for ar in np_batch]


def main(args):
    segmentation_network = SegmentationModule(os.path.expandvars(args.segm_network)).cuda()
    segmentation_network.eval()

    lpips_criterion = LPIPSLossWrapper(args.lpips_network).cuda()
    flow_predictor = FlowPredictor(os.path.expandvars(args.flow_network))

    all_metrics = []
    all_metrics_idx = []

    # load generated images
    gen_frame_paths = list(glob.glob(os.path.join(os.path.expandvars(args.gen_images), '*.jpg')))
    gen_frames_as_img = []
    for fname in gen_frame_paths:
        frame = Image.open(fname).convert('RGB')
        frame_batch = image2batch(frame).cuda() / 2 + 0.5
        assert not (frame_batch < 0).any() and not (frame_batch > 1).any()
        frame_img = batch2pil(frame_batch)[0]
        gen_frames_as_img.append(frame_img)

    # load gt-images, scale, crop and segment
    gt_frame_paths = list(glob.glob(os.path.join(os.path.expandvars(args.gt_images), '*.jpg')))
    gt_frames_as_img = []
    for fname in gt_frame_paths:
        frame = Image.open(fname).convert('RGB')
        frame = frame.resize(get_scale_size(args.resolution, frame.size))

        frame_batch = image2batch(frame).cuda() / 2 + 0.5
        assert not (frame_batch < 0).any() and not (frame_batch > 1).any()
        scaled_size = get_scale_size(args.resolution, frame_batch.shape[2:])
        frame_batch = F.interpolate(frame_batch, size=scaled_size, mode='bilinear', align_corners=False)

        crop_y1, crop_y2, crop_x1, crop_x2 = choose_center_full_size_crop_params(*frame_batch.shape[2:])
        frame_batch = frame_batch[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        frame_img = batch2pil(frame_batch)[0]
        gt_frames_as_img.append(frame_img)

    # compute FID between generated images and gt
    print('Calculating FID for images...')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx]).cuda()
    fid_gt_means, fid_gt_std = _compute_statistics_of_images(gt_frames_as_img, fid_model,
                                                             batch_size=args.batch,
                                                             dims=2048, cuda=True, keep_size=False)
    fid_gen_means, fid_gen_std = _compute_statistics_of_images(gen_frames_as_img, fid_model,
                                                               batch_size=args.batch,
                                                               dims=2048, cuda=True, keep_size=False)
    fid = dict()
    fid['fid_images'] = float(calculate_frechet_distance(fid_gt_means, fid_gt_std, fid_gen_means, fid_gen_std))

    # load generated videos
    for src_path in sorted(glob.glob(os.path.join(args.gen_videos, '*'))):
        if not os.path.isdir(src_path):
            continue
        print(f'Processing {src_path}')
        if src_path.endswith('/'):
            src_path = src_path[:-1]

        vname = os.path.basename(src_path)
        frames = load_video_frames_from_folder(src_path, frame_template=args.frametemplate) / 2 + 0.5
        assert not (frames < 0).any() and not (frames > 1).any()

        # get mask from the first frame
        cur_segm_scores = segmentation_network.predict(frames[:1].cuda(), imgSizes=[args.resolution])
        cur_segm_proba = F.softmax(cur_segm_scores, dim=1)
        
        movable_scores = cur_segm_proba[:, MOVABLE_CLASSES].max(1, keepdim=True)[0]
        immovable_scores = cur_segm_proba[:, [c for c in range(cur_segm_proba.shape[1])
                                                   if c not in MOVABLE_CLASSES]].max(1, keepdim=True)[0]
        shift_mask = (movable_scores > immovable_scores).float()

        print('Flow metrics...')
        flow_metrics = calc_optical_flow_metrics(flow_predictor, frames, shift_mask > 0)

        print('LPIPS metrics...')
        cur_metrics = collections.defaultdict(float)
        lpips = []
        for l in range(1, frames.shape[0], args.batch):
            r = min(l + args.batch, frames.shape[0])
            lpips.append(float(lpips_criterion(frames[l:r].cuda() * (1 - shift_mask), frames[0].cuda() * (1 - shift_mask))))
        cur_metrics['lpips_gen'] = np.mean(lpips)
        sum_dicts(cur_metrics, flow_metrics)

        all_metrics.append(cur_metrics)
        all_metrics_idx.append(vname)

    # load real images, from which the videos were generated, scale, crop and segment
    real_frame_paths = list(glob.glob(os.path.join(os.path.expandvars(args.real_images), '*.jpg')))
    real_frames_as_img = []
    real_frames_with_segm = {}
    for fname in real_frame_paths:
        frame = Image.open(fname).convert('RGB')
        frame = frame.resize(get_scale_size(args.resolution, frame.size))

        # check the interval of stored numbers: 0..1 || -1..1 || 0..255
        frame_batch = image2batch(frame).cuda()
        frame_batch = (frame_batch - frame_batch.min()) / (frame_batch.max() - frame_batch.min())
        assert not (frame_batch < 0).any() and not (frame_batch > 1).any()
        scaled_size = get_scale_size(args.resolution, frame_batch.shape[2:])
        frame_batch = F.interpolate(frame_batch, size=scaled_size, mode='bilinear', align_corners=False)

        crop_y1, crop_y2, crop_x1, crop_x2 = choose_center_full_size_crop_params(*frame_batch.shape[2:])
        frame_batch = frame_batch[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

        frame_img = batch2pil(frame_batch)[0]
        real_frames_as_img.append(frame_img)

        cur_segm_scores = segmentation_network.predict(frame_batch, imgSizes=[args.resolution])
        cur_segm_proba = F.softmax(cur_segm_scores, dim=1)
        f_id = os.path.splitext(os.path.basename(fname))[0]
        real_frames_with_segm[f_id] = (frame_batch, cur_segm_proba)

    # load videos -- animated real images
    animated_frames_by_i = collections.defaultdict(list)

    for src_path in sorted(glob.glob(os.path.join(args.animated_images, '*'))):
        if not os.path.isdir(src_path):
            continue
        print(f'Processing {src_path}')
        if src_path.endswith('/'):
            src_path = src_path[:-1]

        vname = os.path.basename(src_path)
        frames = load_video_frames_from_folder(src_path, frame_template=args.frametemplate) / 2 + 0.5
        assert not (frames < 0).any() and not (frames > 1).any()

        for i, fr in enumerate(batch2pil(frames)):
            animated_frames_by_i[i].append(fr)

        cur_real_frame = None
        cur_real_segm_proba = None
        for frname, (fr, segm) in real_frames_with_segm.items():
            if vname.startswith(frname):
                cur_real_frame = fr
                cur_real_segm_proba = segm
                break
        assert cur_real_frame is not None, (vname, real_frames_with_segm.keys())

        movable_scores = cur_real_segm_proba[:, MOVABLE_CLASSES].max(1, keepdim=True)[0]
        immovable_scores = cur_real_segm_proba[:, [c for c in range(cur_real_segm_proba.shape[1])
                                                   if c not in MOVABLE_CLASSES]].max(1, keepdim=True)[0]
        shift_mask = (movable_scores > immovable_scores).float()

        print('Flow metrics...')
        flow_metrics = calc_optical_flow_metrics(flow_predictor, frames, shift_mask > 0)

        print('LPIPS metrics...')
        cur_metrics = collections.defaultdict(float)
        cur_metrics['lpips_1_frame'] = float(lpips_criterion(frames[:1], cur_real_frame))

        lpips = []
        for l in range(0, frames.shape[0], args.batch):
            r = min(l + args.batch, frames.shape[0])
            lpips.append(float(lpips_criterion(frames[l:r].cuda() * (1 - shift_mask), cur_real_frame.cuda() * (1 - shift_mask))))
        cur_metrics['lpips_anim'] = np.mean(lpips)

        sum_dicts(cur_metrics, flow_metrics)

        all_metrics.append(cur_metrics)
        all_metrics_idx.append(vname)

    print('Calculating FID...')
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    fid_model = InceptionV3([block_idx]).cuda()
    fid_real_means, fid_real_std = _compute_statistics_of_images(real_frames_as_img, fid_model,
                                                                 batch_size=args.batch,
                                                                 dims=2048, cuda=True, keep_size=False)
    for i, cur_gen_frames in animated_frames_by_i.items():
        if i % args.skipframe != 0:
            continue
        cur_fid_means, cur_fid_std = _compute_statistics_of_images(cur_gen_frames, fid_model,
                                                                   batch_size=args.batch,
                                                                   dims=2048, cuda=True, keep_size=False)
        fid[f'fid_{i}'] = float(calculate_frechet_distance(fid_real_means, fid_real_std,
                                                                cur_fid_means, cur_fid_std))

    all_metrics.append(fid)
    all_metrics_idx.append('global_metrics')

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    sum_metrics = pd.DataFrame(all_metrics, index=all_metrics_idx)
    sum_metrics.to_csv(args.outpath, sep='\t')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--outpath', type=str, default='results/metrics.csv', help='Path to file to write metrics to')
    aparser.add_argument('--gen-images', type=str, default='results/generated/256/images', help='Path to generated images')
    aparser.add_argument('--gt-images', type=str, default='results/gt_images', help='Path to gt-images')
    aparser.add_argument('--gen-videos', type=str, default='results/generated/256/noise', 
                         help='Path to generated videos (separate folder with frames for each video)')
    aparser.add_argument('--animated-images', type=str,
                         default='results/encode_and_animate_results/test_images/02_eoif', 
                         help='Path to animated images (separate folder with frames for each video)')
    aparser.add_argument('--real-images', type=str, default='results/test_images', help='Path to real input images')
    aparser.add_argument('--frametemplate', type=str,
                         default='{:05}.jpg',
                         help='Template to generate frame file names')
    aparser.add_argument('--resolution', type=int, default=256, help='Resolution of generated frames')
    aparser.add_argument('--skipframe', type=int, default=10, help='How many frames to skip before evaluating FID')
    aparser.add_argument('--batch', type=int, default=69, help='Batch size for FID and LPIPS calculation')
    aparser.add_argument('--segm-network', type=str,
                         default=os.path.join(constants.RESULT_DIR, 'pretrained_models/ade20k-resnet50dilated-ppm_deepsup'),
                         help='Path to ade20k-resnet50dilated-ppm_deepsup')
    aparser.add_argument('--flow-network', type=str,
                         default=os.path.join(constants.RESULT_DIR, 'pretrained_models/SuperSloMo.ckpt'),
                         help='Path to SuperSloMo.ckpt')
    aparser.add_argument('--lpips-network', type=str,
                         default=os.path.join(constants.RESULT_DIR, 'pretrained_models/lpips_models/vgg.pth'),
                         help='Path to vgg.pth')

    main(aparser.parse_args())

