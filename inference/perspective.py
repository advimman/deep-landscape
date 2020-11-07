import glob
import os

import PIL.Image as Image
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from kornia import convert_points_to_homogeneous, convert_points_from_homogeneous
from skimage.transform import ProjectiveTransform, warp as skimage_warp

from inference.base_image_utils import image2batch, get_scale_size, choose_center_full_size_crop_params
from inference.metrics.slomo.flow import SloMoFlow


def load_video_frames_from_folder(dirname, skip_frame=1, frame_template='{}.jpg', target_size=256):
    n_frames = len(os.listdir(dirname))
    result = []
    for frame_i in range(0, n_frames, skip_frame):
        frame = Image.open(os.path.join(dirname, frame_template.format(frame_i)))

        frame = image2batch(frame)
        scaled_size = get_scale_size(target_size, frame.shape[2:])
        frame = F.interpolate(frame, size=scaled_size, mode='bilinear', align_corners=False)

        crop_y1, crop_y2, crop_x1, crop_x2 = choose_center_full_size_crop_params(*frame.shape[2:])
        frame = frame[0, :, crop_y1:crop_y2, crop_x1:crop_x2].numpy().astype('float32')

        result.append(frame)
    return torch.from_numpy(np.stack(result))


class FlowPredictor:
    def __init__(self, flownet_path, batch_size=1, device='cuda'):
        self.device = torch.device(device)
        self.flownet = SloMoFlow(model_path=flownet_path).to(self.device)
        self.flownet.eval()
        self.batch_size = batch_size

    def predict_flow(self, frames):
        frames_num, channels, height, width = frames.shape
        frames = frames.to(self.device)

        with torch.no_grad():
            frames = ((frames / 2 + 0.5) * 255).clamp(0, 255)

            frames_upscaled = F.interpolate(frames, size=(1024, 1024), mode='bilinear', align_corners=True)
            frames_left = frames_upscaled[:-1]
            frames_right = frames_upscaled[1:]

            forward_flow = []
            backward_flow = []
            for batch_start in range(0, frames_left.shape[0], self.batch_size):
                frames_left_chunk = frames_left[batch_start:batch_start + self.batch_size]
                frames_right_chunk = frames_right[batch_start:batch_start + self.batch_size]
                (forward_flow_flat_chunk,
                 backward_flow_flat_chunk) = self.flownet._flow_estimation(frames_left_chunk.clone(),
                                                                           frames_right_chunk.clone())
                forward_flow.append(forward_flow_flat_chunk)
                backward_flow.append(backward_flow_flat_chunk)

            forward_flow = torch.cat(forward_flow, dim=0)
            backward_flow = torch.cat(backward_flow, dim=0)

            forward_flow = forward_flow / 512
            forward_flow = F.interpolate(forward_flow, size=(height, width), mode='bilinear', align_corners=False)

            backward_flow = backward_flow / 512
            backward_flow = F.interpolate(backward_flow, size=(height, width), mode='bilinear', align_corners=False)

            return forward_flow, backward_flow


def get_horizon_line_coords(segmentation_logits, below_sky=10, sky_class=2):
    batch_size, channels, height, _ = segmentation_logits.shape
    batch_idx, vert_idx, _ = torch.nonzero(segmentation_logits.argmax(1) == sky_class, as_tuple=True)
    result = []
    for sample_i in range(batch_size):
        sample_mask = batch_idx == sample_i
        if not sample_mask.any():
            cur_result = 0
        else:
            max_coord = vert_idx[sample_mask].max()
            cur_result = int(min(height, max_coord + below_sky)) / height
        result.append(cur_result)
    return result


def get_shifts_from_sky_hom(hom, horizon_line):
    hom = torch.from_numpy(hom).float().unsqueeze(0)
    horizon_line = horizon_line * 2 - 1
    points_src = torch.tensor([[[-1,           -1],  # left top
                                [ 1,           -1],  # right top
                                [-1, horizon_line],  # left bottom
                                [ 1, horizon_line]]]).float()  # right bottom
    points_src_hom = convert_points_to_homogeneous(points_src)
    points_tgt_hom = (hom @ points_src_hom.transpose(1, 2)).transpose(1, 2)
    points_tgt = convert_points_from_homogeneous(points_tgt_hom)

    shifts_flat = points_tgt - points_src
    shifts = shifts_flat.view(2, 2, 2)

    return shifts.numpy()


def make_homography(shifts, horizon_line, resolution):
    horizon_line = resolution * horizon_line
    left_top_dx, left_top_dy = shifts[0][0]
    right_top_dx, right_top_dy = shifts[0][1]
    left_hor_dx, left_hor_dy = shifts[1][0]
    right_hor_dx, right_hor_dy = shifts[1][1]
    points_src = np.array([[0, 0],  # left top
                           [resolution, 0],  # right top
                           [0, horizon_line],  # left bottom
                           [resolution, horizon_line]], dtype='float32')  # right bottom
    points_tgt = np.array([[left_top_dx, left_top_dy],  # left top
                           [resolution + right_top_dx, right_top_dy],  # right top
                           [left_hor_dx, horizon_line + left_hor_dy],  # left bottom
                           [resolution + right_hor_dx, horizon_line + right_hor_dy]], dtype='float32')  # right bottom
    sky_transform = ProjectiveTransform()
    sky_transform.estimate(points_src, points_tgt)

    points_src = np.array([[0, horizon_line],  # left horizon
                           [resolution, horizon_line],  # right horizon
                           [0, resolution],  # left bottom
                           [resolution, resolution]], dtype='float32')  # right bottom
    points_tgt = np.array([[left_hor_dx, horizon_line - left_hor_dy],  # left horizon
                           [resolution + right_hor_dx, horizon_line - right_hor_dy],  # right horizon
                           [left_top_dx, resolution - left_top_dy],  # left bottom
                           [resolution + right_top_dx, resolution - right_top_dy]], dtype='float32')  # right bottom
    earth_transform = ProjectiveTransform()
    earth_transform.estimate(points_src, points_tgt)

    return sky_transform, earth_transform


def make_homography_kornia(shifts, horizon_line):
    horizon_line = horizon_line * 2 - 1

    left_top_dx, left_top_dy = shifts[0][0]
    right_top_dx, right_top_dy = shifts[0][1]
    left_hor_dx, left_hor_dy = shifts[1][0]
    right_hor_dx, right_hor_dy = shifts[1][1]

    if horizon_line > -1 + 1e-4:
        points_src = torch.tensor([[[-1, -1],  # left top
                                    [ 1, -1],  # right top
                                    [-1, horizon_line],  # left bottom
                                    [ 1, horizon_line]]]).float()  # right bottom
        points_tgt = torch.tensor([[[-1 + left_top_dx,  -1 + left_top_dy],  # left top
                                    [ 1 + right_top_dx, -1 + right_top_dy],  # right top
                                    [-1 + left_hor_dx,   horizon_line + left_hor_dy],  # left bottom
                                    [ 1 + right_hor_dx,  horizon_line + right_hor_dy]]]).float()  # right bottom
        sky_transform = kornia.get_perspective_transform(points_src, points_tgt)
    else:
        points_src = torch.tensor([[[-1, -1],  # left top
                                    [1, -1],  # right top
                                    [-1, 1],  # left bottom
                                    [1,  1]]]).float()  # right bottom
        sky_transform = kornia.get_perspective_transform(points_src, points_src)

    if horizon_line <= 1 - 1e-4:
        points_src = torch.tensor([[[-1, horizon_line],  # left horizon
                                    [ 1, horizon_line],  # right horizon
                                    [-1, 1],  # left bottom
                                    [ 1, 1]]]).float()  # right bottom
        points_tgt = torch.tensor([[[-1 + left_hor_dx, horizon_line - left_hor_dy],  # left horizon
                                    [ 1 + right_hor_dx, horizon_line - right_hor_dy],  # right horizon
                                    [-1 + left_top_dx, 1 - left_top_dy],  # left bottom
                                    [ 1 + right_top_dx, 1 - right_top_dy]]]).float()  # right bottom
        earth_transform = kornia.get_perspective_transform(points_src, points_tgt)
    else:
        points_src = torch.tensor([[[-1, -1],  # left horizon
                                    [ 1, -1],  # right horizon
                                    [-1,  1],  # left bottom
                                    [ 1,  1]]]).float()  # right bottom
        earth_transform = kornia.get_perspective_transform(points_src, points_src)

    return sky_transform, earth_transform


def warp_homography(images, transforms, n_iter=1, horizon_line=None):
    if horizon_line is None:
        horizon_line = 0.66

    horizon_line = int(images.shape[-2] * horizon_line)

    sky_transform, earth_transform = transforms
    if n_iter < 0:
        sky_transform = ProjectiveTransform(sky_transform._inv_matrix)
        earth_transform = ProjectiveTransform(earth_transform._inv_matrix)
        n_iter = -n_iter

    sum_sky_transform = sky_transform
    sum_earth_transform = earth_transform
    for _ in range(int(n_iter - 1)):
        sum_sky_transform = sum_sky_transform + sky_transform
        sum_earth_transform = sum_earth_transform + earth_transform

    images_np = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    result = []
    for img in images_np:
        sky_warped = skimage_warp(img, sum_sky_transform.inverse, mode='wrap', order=3)
        earth_warped = skimage_warp(img, sum_earth_transform.inverse, mode='wrap', order=3)
        cur_image = np.concatenate((sky_warped[:horizon_line], earth_warped[horizon_line:]), axis=0)
        result.append(cur_image)
    result = np.stack(result, axis=0)
    result = np.transpose(result, (0, 3, 1, 2))
    result = torch.from_numpy(result).to(images)
    return result


def warp_homography_kornia(images, transforms, n_iter=1, horizon_line=None):
    if horizon_line is None:
        horizon_line = 1

    height, width = images.shape[-2:]
    horizon_line_px = int(height * horizon_line)

    sky_transform, earth_transform = transforms
    if n_iter < 0:
        sky_transform = torch.inverse(sky_transform)
        earth_transform = torch.inverse(earth_transform)
        n_iter = -n_iter

    sky_transform = sky_transform.to(images.device)
    earth_transform = earth_transform.to(images.device)

    sum_sky_transform = sky_transform
    sum_earth_transform = earth_transform
    for _ in range(int(n_iter - 1)):
        sum_sky_transform = sum_sky_transform @ sky_transform
        sum_earth_transform = sum_earth_transform @ earth_transform

    warper = kornia.geometry.warp.HomographyWarper(height, width, padding_mode='reflection')
    if 1e-4 <= horizon_line <= 1 - 1e-4:
        sky_warped = warper(images, sum_sky_transform)
        earth_warped = warper(images, sum_earth_transform)
        result = torch.cat((sky_warped[:, :, :horizon_line_px], earth_warped[:, :, horizon_line_px:]), dim=2)
    elif horizon_line < 1e-4:
        result = warper(images, sum_earth_transform)
    elif 1 - 1e-4 < horizon_line:
        result = warper(images, sum_sky_transform)

    return result


def make_manual_homography(dx, dy, horizon_line, resolution, yscale=2, rscale=1.5):
    shifts = [[[dx * yscale, dy * yscale], [dx * yscale, dy * yscale * rscale]],
              [[dx,          dy],          [dx,          dy * rscale]]]
    return make_homography(shifts, horizon_line, resolution)


def make_manual_shifts(dx, dy, yscale=2, rscale=1.5):
    shifts = [[[dx * yscale, dy * yscale], [dx * yscale, dy * yscale * rscale]],
              [[dx,          dy],          [dx,          dy * rscale]]]
    return shifts


def make_manual_homography_kornia(dx, dy, horizon_line, yscale=2, rscale=1.5):
    shifts = make_manual_shifts(dx, dy, yscale=yscale, rscale=rscale)
    return make_homography_kornia(shifts, horizon_line)


def save_hom_shifts(hom, fname):
    np.savetxt(fname, hom.reshape((4, 2)))


def load_hom_shifts(fname):
    return np.loadtxt(fname).reshape((2, 2, 2))


class RandomHomography:
    def __init__(self, indir):
        print(indir)
        self.all_shifts = [(self.get_shift_id(fname), load_hom_shifts(fname))
                           for fname in glob.glob(os.path.join(os.path.expandvars(indir), '*.csv'))]

    def __call__(self, horizon_line):
        i = np.random.randint(len(self.all_shifts))
        cur_id, cur_shifts = self.all_shifts[i]
        return cur_id, make_homography_kornia(cur_shifts, horizon_line)

    def get_shift_id(self, fname):
        return os.path.splitext(os.path.basename(fname))[0]
