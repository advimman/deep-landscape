import os
from io import BytesIO
import random

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

import constants
import tensor_transforms as tt
from logger import LOGGER
from utils import format_for_lmdb


def get_img_from_lmdb(txn, *key_parts):
    key = format_for_lmdb(*key_parts)
    img_bytes = txn.get(key)
    buf = BytesIO(img_bytes)
    img = Image.open(buf)
    return img


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=constants.INIT_SIZE):
        self.path = path
        self.transform = transform
        self.resolution = resolution

    def __len__(self):
        return self.length

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        env = getattr(self, 'env', None)
        if env is not None:
            env.close()
        self._resolution = resolution
        self.crop = RandomCrop(self.resolution)
        path = os.path.join(self.path, str(self.resolution))
        self.env = lmdb.open(
            path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get(format_for_lmdb('length')).decode('utf-8'))


class MultiResolutionImageDataset(MultiResolutionDataset):
    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            img = get_img_from_lmdb(txn, index)
        out = self.transform(self.crop(img)).unsqueeze_(0)
        return out


class MultiResolutionMultiFrameDataset(MultiResolutionDataset):
    def __init__(self, path, transform, tensor_transform, num_frames, resolution=constants.INIT_SIZE):
        super().__init__(path, transform, resolution)
        self.num_frames = num_frames
        self.tensor_transform = tensor_transform

    @MultiResolutionDataset.resolution.setter
    def resolution(self, resolution):
        super(MultiResolutionMultiFrameDataset, type(self)).resolution.fset(self, resolution)
        self.crop = tt.RandomCrop(self.resolution)

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            try:
                length = int(txn.get(format_for_lmdb(index, 'length')).decode('utf-8'))
            except AttributeError:
                LOGGER.warning(f'A video with an index {index} has a length None.')
                return self.__getitem__(random.randint(0, len(self) - 1))

        if length < self.num_frames:
            LOGGER.warning(f'There is only {length} frames in a video with an index {index}, '
                           'so a random video will be used instead.')
            return self.__getitem__(random.randint(0, len(self) - 1))

        selected_frame_indexes = random.sample(range(length), self.num_frames)
        selected_frames = []
        with env.begin(write=False) as txn:
            for i in selected_frame_indexes:
                img = get_img_from_lmdb(txn, index, i)
                selected_frames.append(self.transform(img))
        for frame in selected_frames[1:]:
            if frame.shape != selected_frames[0].shape:
                LOGGER.warning(f'Frames in a video with an index {index} have different sizes, '
                               'so a random video will be used instead.')
                return self.__getitem__(random.randint(0, len(self) - 1))
        frames = self.tensor_transform(self.crop(torch.stack(selected_frames)))
        return frames


class MultiResolutionMultiCropDataset(MultiResolutionMultiFrameDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            length = int(txn.get(format_for_lmdb(index, 'length')).decode('utf-8'))
        if length < 1:
            LOGGER.warning(f'There is no frames in a video with an index {index}, '
                           'so a random video will be used instead.')
            return self.__getitem__(random.randint(0, len(self) - 1))

        selected_frame_index = random.randint(0, length - 1)
        with self.env.begin(write=False) as txn:
            img = get_img_from_lmdb(txn, index, selected_frame_index)
        transformed_frame = self.transform(img).unsqueeze_(0)
        crops = self.tensor_transform(torch.cat([self.crop(transformed_frame) for i in range(self.num_frames)]))
        return crops
