"""Modified from https://github.com/CSAILVision/semantic-segmentation-pytorch"""

import os
import sys

import numpy as np
import torch

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def color_encode(labelmap, colors, mode='RGB'):
    if isinstance(labelmap, torch.Tensor):
        if labelmap.ndim > 3:
            labelmap = labelmap[0, 0, ...]
        labelmap = labelmap.cpu().numpy()

    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def color_encode_torch(labelmap: torch.Tensor, colors: np.ndarray, unsqueeze=False):
    device = labelmap.device
    labelmap_rgb = torch.zeros((labelmap.shape[0], 3, *labelmap.shape[-2:]), device=device)
    for label in labelmap.unique().cpu().int().numpy():
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label).float() * torch.tensor(np.tile(colors[label], (labelmap.shape[-1], labelmap.shape[-2], 1))).permute(2, 0, 1).float().to(device)

        if False:  # TODO:
            labelmap_rgb[labelmap == label] = torch.from_numpy(colors[label])[None, 3, None, None].to(labelmap_rgb.device)

    return labelmap_rgb
