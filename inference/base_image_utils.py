import numpy as np
import torch


def get_scale_size(target_size, cur_size):
    cur_size = np.array(cur_size)
    factor = max(target_size / cur_size)
    return tuple((cur_size * factor).astype(int))


def choose_center_full_size_crop_params(height, width):
    if width < height:
        offset_x = 0
        offset_y = (height - width) // 2
        size = width
    else:
        offset_y = 0
        offset_x = (width - height) // 2
        size = height
    return offset_y, offset_y + size, offset_x, offset_x + size


def image2batch(image):
    arr = np.transpose(np.array(image), (2, 0, 1)).astype('float32')
    arr /= 255
    arr -= 0.5
    arr *= 2
    arr = torch.from_numpy(arr).unsqueeze(0)
    return arr


def batch2array(batch):
    return batch[0].detach().cpu().numpy()
