#!/usr/bin/env python3

import collections
import copy
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
import scipy.interpolate
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from easydict import EasyDict as edict

from inference.base_image_utils import get_scale_size, choose_center_full_size_crop_params, batch2array, image2batch
from inference.datasets import expand_latents, get_shape
from inference.fine_tune_pipeline import optimize_latents, fine_tune_generator
from inference.inference_utils import get_noise_for_infer, sum_dicts
from inference.metrics import LPIPSLossWrapper, SSIM
from inference.perspective import get_horizon_line_coords, make_manual_homography_kornia, warp_homography_kornia, \
    RandomHomography
from inference.segmentation import SegmentationModule
from inference.train_encoder import DecoderFromLatents
from utils import get_mean_style
import constants


def noise_cycle_shift(latents, part, projective_transforms,
                      shift_names=None, shift_channels=None, rescale_after_shift=False,
                      shift_in_hr=False, horizon_line=None):
    latents = copy.deepcopy(latents)
    if shift_names is None:
        shift_names = list(latents.keys())

    if shift_channels is None:
        shift_channels = (-1,)

    for name in shift_names:
        if not name.startswith('noise'):
            continue

        if shift_in_hr:
            orig_size_lr = latents[name].shape[-2:]
            latents[name] = F.interpolate(latents[name].squeeze(1), size=(shift_in_hr, shift_in_hr),
                                          mode='bicubic', align_corners=False).unsqueeze(1)

        before = latents[name][:, 0, shift_channels]
        before_mean = before.mean()
        before_std = before.std()
        after = warp_homography_kornia(before, projective_transforms,
                                       n_iter=part, horizon_line=horizon_line).unsqueeze(1)
        if rescale_after_shift:
            after = (after - after.mean()) / after.std() * before_std + before_mean
        latents[name][:, 0, shift_channels] = after

        if shift_in_hr:
            latents[name] = F.interpolate(latents[name].squeeze(1), size=orig_size_lr,
                                          mode='bicubic', align_corners=False).unsqueeze(1)
    return latents


def rescale_img_tensor(tensor, out_size):
    return F.interpolate(tensor.unsqueeze(0), size=out_size, mode='bilinear', align_corners=False)[0]


def gen_images_cycle_shift(latents, decoder, steps=10, shift_names=None, shift_channels=None, rescale_after_shift=False,
                           min_shift=0, max_shift=2, animate_w_names=(), target_z_func=None, projective_transforms=None,
                           shift_in_hr=False, horizon_line=None):
    images = []
    all_latents = []

    latents_for_shift = copy.deepcopy(latents)

    if target_z_func is not None:
        all_times = np.linspace(0, 1, steps)
        z_interpolations = {name: target_z_func(latents_for_shift[name], all_times) for name in animate_w_names}

    for step_i, shift in enumerate(np.linspace(min_shift, max_shift, steps)):
        new_latents = copy.deepcopy(latents_for_shift)

        new_latents = noise_cycle_shift(new_latents, shift, projective_transforms=projective_transforms,
                                        shift_names=shift_names, shift_channels=shift_channels,
                                        rescale_after_shift=rescale_after_shift, shift_in_hr=shift_in_hr,
                                        horizon_line=horizon_line)

        if target_z_func is not None:
            for key in animate_w_names:
                new_latents[key] = z_interpolations[key][step_i]

        all_latents.append(new_latents)
        new_img = batch2array(decoder(new_latents))[0]
        images.append(new_img)
    return images


ZTimeStep = collections.namedtuple('ZTimeStep', 'time z'.split(' '))


class SplineStyleAnimation:
    def __init__(self, mlp_approximator, *steps, **spline_kwargs):
        self.mlp_approximator = mlp_approximator
        self.steps = steps
        self.spline_kwargs = spline_kwargs

    def __call__(self, styles, new_times):
        with torch.no_grad():
            intermediate_points = []
            for step in self.steps:
                cur_data = torch.cat((styles,
                                      torch.tensor(step.z).to(styles)[None, None, ...]),
                                     dim=-1)
                intermediate_points.append(self.mlp_approximator(cur_data))
            intermediate_points_flat = torch.stack(intermediate_points).view(-1, styles.shape[-1]).detach().cpu().numpy()
            times = [step.time for step in self.steps]
            new_styles_flat = scipy.interpolate.make_interp_spline(times, intermediate_points_flat, **self.spline_kwargs)(new_times)
            new_styles = torch.from_numpy(new_styles_flat).to(styles.device).view(len(new_times), *styles.shape).float()
            return new_styles


def write_video(out_path, frames, fps=24, write_frames=False):
    channels, height, width = frames[0].shape

    if write_frames:
        frames_dirname = out_path + '_frames'
        os.makedirs(frames_dirname, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    try:
        for i, frame in enumerate(frames):
            frame = np.array(frame)
            frame /= 2
            frame += 0.5
            frame *= 255
            frame = np.clip(frame, 0, 255)
            frame = frame[[2, 1, 0]]
            frame = np.transpose(frame, (1, 2, 0)).astype('uint8')
            writer.write(frame)
            if write_frames:
                cv2.imwrite(os.path.join(frames_dirname, f'{i:05d}.jpg'), frame)
    finally:
        writer.release()


def calc_segmentation_posterior_error(segm_model, target_segm, animated_frames,
                                      still_segm_mask, first_frame, lpips_model, ssim_model, **predict_kwargs):
    with torch.no_grad():
        result = collections.defaultdict(float)
        discrete_target = target_segm.argmax(dim=1)

        first_frame_still = first_frame * still_segm_mask

        for frame_i, cur_frame in enumerate(animated_frames):
            cur_frame = torch.from_numpy(cur_frame).cuda().unsqueeze(0)
            cur_segm = segm_model.predict(cur_frame, **predict_kwargs)

            cur_segm_discrete = cur_segm.argmax(dim=1)
            result[f'acc_{frame_i}'] = float((discrete_target == cur_segm_discrete).float().mean())

            cur_frame_still = cur_frame * still_segm_mask
            result[f'lpips_{frame_i}'] = float(lpips_model(cur_frame_still, first_frame_still).mean())
            result[f'ssim_{frame_i}'] = float(ssim_model(cur_frame_still, first_frame_still).mean())

        return result


def main(args):
    with open(args.config) as f:
        config = edict(yaml.load(f, Loader=yaml.SafeLoader))

    os.makedirs(args.outdir, exist_ok=True)

    if config.encoder_checkpoint is None or config.encoder_checkpoint.lower() == 'none':
        encoder = None
    else:
        encoder = torch.load(os.path.join(constants.RESULT_DIR, config.encoder_checkpoint)).cuda()

    decoder = DecoderFromLatents(**config.decoder_kwargs)
    target_size = decoder.infer_model['resolution']

    if config.segmentation:
        config.segmentation.module_kwargs['models_dirname'] = os.path.join(
            constants.RESULT_DIR, config.segmentation.module_kwargs['models_dirname'])
        segmentation_network = SegmentationModule(**config.segmentation.module_kwargs).cuda()
        segmentation_network.eval()
    else:
        segmentation_network = None

    if 'target_z_func' in config.shift_kwargs:
        mlp_approx_model = torch.load(os.path.join(
            constants.RESULT_DIR, config.shift_kwargs.target_z_func.mlp_approx_model)).cuda()
        target_z_func_kwargs = config.shift_kwargs.target_z_func.kwargs
        steps = config.shift_kwargs.target_z_func.steps
        config.shift_kwargs.target_z_func = SplineStyleAnimation(mlp_approx_model,
                                                                 *steps,
                                                                 **target_z_func_kwargs)

    homography_kwargs = args.homography_dir

    if 'num_real_homs_per_image' in config.shift_kwargs:
        num_real_homs_per_image = config.shift_kwargs.pop('num_real_homs_per_image')
        random_hom = RandomHomography(homography_kwargs)
    else:
        num_real_homs_per_image = 0

    full_output = config.get('full_output', True)
    save_frames_as_jpg = full_output or config.get('save_frames_as_jpg', True)
    calc_metrics = config.get('calc_metrics', False)
    infer_using_mask = config.get('infer_using_mask', False)
    fine_tune_generator_using_mask = config.get('fine_tune_generator_using_mask', False)

    if calc_metrics:
        lpips_criterion = LPIPSLossWrapper(model_path=os.path.join(
            constants.RESULT_DIR, config.get('lpips_model_path', None))).cuda()
        ssim_criterion = SSIM().cuda()
        sum_metrics = []
        sum_metrics_idx = []

    for src_path in sorted(glob.glob(args.inglob)):
        print()
        print('Animating', src_path)
        fname = os.path.splitext(os.path.basename(src_path))[0]

        src_image = Image.open(src_path).convert('RGB')
        src_image = src_image.resize(get_scale_size(config.max_in_resolution, src_image.size))

        img_batch_orig = image2batch(src_image).cuda()
        scaled_size = get_scale_size(target_size, img_batch_orig.shape[2:])
        img_batch_scaled = F.interpolate(img_batch_orig, size=scaled_size, mode='bilinear', align_corners=False)

        crop_y1, crop_y2, crop_x1, crop_x2 = choose_center_full_size_crop_params(*img_batch_scaled.shape[2:])
        img_batch_cropped = img_batch_scaled[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
        img_batch_cropped01 = img_batch_cropped / 2 + 0.5

        config.shift_kwargs['horizon_line'] = None

        with torch.no_grad():
            shift_mask = None
            if segmentation_network is not None:
                img_batch_for_segm = img_batch_orig / 2 + 0.5
                cls_scores = segmentation_network.predict(img_batch_for_segm, **config.segmentation.predict_kwargs)
                cls_scores = F.interpolate(cls_scores, size=scaled_size, mode='bilinear', align_corners=False)
                cls_scores = cls_scores[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

                cls_proba = F.softmax(cls_scores, dim=1)

                config.shift_kwargs['horizon_line'] = get_horizon_line_coords(cls_scores)[0]  # if infer_using_mask else 1

                movable_scores = cls_scores[:, config.segmentation.movable_classes].max(1, keepdim=True)[0]
                cls_scores[:, config.segmentation.movable_classes] = 0
                immovable_scores = cls_scores.max(1, keepdim=True)[0]

                shift_mask = (movable_scores > immovable_scores).float()

                shift_mask_np = shift_mask.detach().cpu().numpy()[0, 0]
                if config.segmentation.erode > 0:
                    shift_mask_np = cv2.erode(shift_mask_np, dilation_kernel)
                shift_mask = torch.from_numpy(shift_mask_np).to(shift_mask)[None, None, ...]
            else:
                config.shift_kwargs['horizon_line'] = 1

            if homography_kwargs is not None:
                if num_real_homs_per_image == 0:
                    homography_kwargs = copy.deepcopy(homography_kwargs)
                    homography_kwargs['horizon_line'] = config.shift_kwargs['horizon_line']
                    config.shift_kwargs['projective_transforms'] = make_manual_homography_kornia(**homography_kwargs)
                else:
                    hom_id, hom = random_hom(config.shift_kwargs['horizon_line'])
                    config.shift_kwargs['projective_transforms'] = hom

            if encoder is None:
                mean_style = get_mean_style(decoder.infer_model['g_running'], 'cuda', 512)
                latents = {f'latent_wprime:{level_i}:{j}': mean_style.clone().detach().unsqueeze(0)
                           for level_i in range(decoder.infer_model['step'] + 1)
                           for j in range(2)}
            else:
                latents = encoder(img_batch_cropped)

                if config.get('take_only_latents', None):
                    latents = {name: var for name, var in latents.items()
                               if re.search(config['take_only_latents'], name)}

                for name in list(latents):
                    latents[name] = latents[name].unsqueeze(1)

        noise = get_noise_for_infer(decoder.infer_model.g_running, batch_size=1, step=decoder.infer_model.step,
                                    scale=config.get('init_noise_scale', 1))
        noise = expand_latents(noise, name_prefix='noise')
        for name, var in noise.items():
            if name not in latents:
                latents[name] = var

        latents_for_encoder_vis = copy.deepcopy(latents)
        latents_for_encoder_vis.update(
            expand_latents(get_noise_for_infer(decoder.infer_model.g_running, batch_size=1,
                                               step=decoder.infer_model.step),
                           name_prefix='noise')
        )

        encoder_image_tensor = decoder(latents_for_encoder_vis)
        encoder_image = batch2array(encoder_image_tensor)[0]
        encoder_image_tensor01 = encoder_image_tensor / 2 + 0.5

        if full_output or calc_metrics:
            encoder_frames = [encoder_image]
            encoder_frames.extend(gen_images_cycle_shift(latents_for_encoder_vis, decoder,
                                                         **config.shift_kwargs))

        if calc_metrics:
            cur_metrics = collections.defaultdict(float)
            cur_metrics.update(dict(lpips_1_enc=float(lpips_criterion(encoder_image_tensor01.squeeze(1),
                                                                      img_batch_cropped01).mean()),
                                    ssim_1_enc=float(ssim_criterion(encoder_image_tensor01.squeeze(1),
                                                                    img_batch_cropped01).mean())))
            if segmentation_network is not None:
                sum_dicts(cur_metrics,
                          calc_segmentation_posterior_error(segmentation_network,
                                                            cls_proba,
                                                            [fr / 2 + 0.5 for fr in encoder_frames],
                                                            still_segm_mask=1 - shift_mask,
                                                            first_frame=img_batch_cropped01,
                                                            lpips_model=lpips_criterion,
                                                            ssim_model=ssim_criterion),
                          prefix='segm_1_enc')

        latents = optimize_latents(img_batch_cropped, latents, decoder,
                                   still_segm_mask=(1 - shift_mask) if infer_using_mask else None,
                                   **config.fine_tune_kwargs)
        real_image_cropped = batch2array(img_batch_cropped)
        tuned_image_tensor = decoder(latents)
        tuned_image_tensor01 = tuned_image_tensor / 2 + 0.5
        tuned_image = batch2array(tuned_image_tensor)[0]

        if full_output or calc_metrics:
            frames = [tuned_image]
            frames.extend(gen_images_cycle_shift(latents, decoder,
                                                 **config.shift_kwargs))

        if calc_metrics:
            cur_metrics.update(dict(lpips_2_opt=float(lpips_criterion(tuned_image_tensor01.squeeze(1),
                                                                      img_batch_cropped01).mean()),
                                    ssim_2_opt=float(ssim_criterion(tuned_image_tensor01.squeeze(1),
                                                                    img_batch_cropped01).mean())))
            if segmentation_network is not None:
                sum_dicts(cur_metrics,
                          calc_segmentation_posterior_error(segmentation_network,
                                                            cls_proba,
                                                            [fr / 2 + 0.5 for fr in frames],
                                                            still_segm_mask=1 - shift_mask,
                                                            first_frame=img_batch_cropped01,
                                                            lpips_model=lpips_criterion,
                                                            ssim_model=ssim_criterion),
                          prefix='segm_2_opt')

        tuned_decoder = fine_tune_generator(latents, img_batch_cropped, decoder,
                                            still_segm_mask=(1 - shift_mask) if fine_tune_generator_using_mask else None,
                                            **config.generator_fine_tune_kwargs)[0]
        tuned2_image_tensor = tuned_decoder(latents)
        tuned2_image_tensor01 = tuned2_image_tensor / 2 + 0.5
        tuned2_image = batch2array(tuned2_image_tensor)[0]

        if calc_metrics:
            cur_metrics.update(dict(lpips_3_ft=float(lpips_criterion(tuned2_image_tensor01.squeeze(1),
                                                                     img_batch_cropped01).mean()),
                                    ssim_3_ft=float(ssim_criterion(tuned2_image_tensor01.squeeze(1),
                                                                   img_batch_cropped01).mean())))

        if num_real_homs_per_image > 0 and homography_kwargs is not None:
            used_homs = set()
            actual_homs_n = 0
            for _ in range(num_real_homs_per_image):
                found_new_hom = False
                for _ in range(1000):
                    hom_id, hom = random_hom(config.shift_kwargs['horizon_line'])
                    if hom_id not in used_homs:
                        used_homs.add(hom_id)
                        found_new_hom = True
                        break
                if not found_new_hom:
                    break

                actual_homs_n += 1
                config.shift_kwargs['projective_transforms'] = hom
                tuned_frames = [tuned2_image.copy()]
                tuned_frames.extend(gen_images_cycle_shift(latents, tuned_decoder,
                                                           **config.shift_kwargs))

                if calc_metrics and segmentation_network is not None:
                    sum_dicts(cur_metrics,
                              calc_segmentation_posterior_error(segmentation_network,
                                                                cls_proba,
                                                                [fr / 2 + 0.5 for fr in tuned_frames],
                                                                still_segm_mask=1 - shift_mask,
                                                                first_frame=img_batch_cropped01,
                                                                lpips_model=lpips_criterion,
                                                                ssim_model=ssim_criterion),
                              prefix='segm_3_ft')

                if full_output:
                    frames = [np.concatenate((np.concatenate((real_image_cropped, encoder_image, enc_frame), axis=2),
                                              np.concatenate((real_image_cropped, tuned_image, frame), axis=2),
                                              np.concatenate((real_image_cropped, tuned2_image, frame2), axis=2)),
                                             axis=1)
                              for enc_frame, frame, frame2 in zip(encoder_frames, frames, tuned_frames)]
                else:
                    frames = tuned_frames

                write_video(os.path.join(args.outdir, f'{fname}_hom{hom_id}.avi'), frames,
                            write_frames=save_frames_as_jpg, **config.video_kwargs)

            if calc_metrics and segmentation_network is not None:
                for k in list(cur_metrics):
                    if k.startswith('segm_3_ft'):
                        cur_metrics[k] /= actual_homs_n
        else:
            tuned_frames = [tuned2_image]
            tuned_frames.extend(gen_images_cycle_shift(latents, tuned_decoder,
                                                       **config.shift_kwargs))

            if full_output:
                frames = [np.concatenate((np.concatenate((real_image_cropped, encoder_image, enc_frame), axis=2),
                                          np.concatenate((real_image_cropped, tuned_image, frame), axis=2),
                                          np.concatenate((real_image_cropped, tuned2_image, frame2), axis=2)),
                                         axis=1)
                          for enc_frame, frame, frame2 in zip(encoder_frames, frames, tuned_frames)]
            else:
                frames = tuned_frames

            write_video(os.path.join(args.outdir, fname + '.avi'), frames, write_frames=save_frames_as_jpg,
                        **config.video_kwargs)

        if calc_metrics:
            sum_metrics.append(cur_metrics)
            sum_metrics_idx.append(fname)

        if segmentation_network is not None:
            del shift_mask
            del cls_scores
            del movable_scores
            del immovable_scores
            del latents
        torch.cuda.empty_cache()

    if calc_metrics:
        sum_metrics = pd.DataFrame(sum_metrics, index=sum_metrics_idx)
        sum_metrics.to_csv(os.path.join(args.outdir, f'metrics{args.suffix}.tsv'), sep='\t')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config')
    aparser.add_argument('inglob')
    aparser.add_argument('outdir')
    aparser.add_argument('homography_dir')
    aparser.add_argument('--suffix', type=str, default='', help='Suffix to metrics filename')

    args = aparser.parse_args()
    main(args)
