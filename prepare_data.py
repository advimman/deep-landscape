#!/usr/bin/env python3

from glob import glob
import os
from io import BytesIO
import argparse
import multiprocessing

import cv2
import lmdb
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as trans_fn

import constants
from utils import format_for_lmdb


class Resizer:
    def __init__(self, data_type, *, size, quality):
        assert data_type in ('images', 'videos'), data_type
        self.data_type = data_type
        self.size = size
        self.quality = quality

    def get_resized_bytes(self, img):
        img = trans_fn.resize(img, self.size)
        buf = BytesIO()
        img.save(buf, format='jpeg', quality=self.quality)
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        if self.data_type == 'images':
            img = Image.open(filename)
            img = img.convert('RGB')
            return self.get_resized_bytes(img)

        elif self.data_type == 'videos':
            frames = []
            cap = cv2.VideoCapture(filename)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame)
                    img_bytes = self.get_resized_bytes(img_pil)
                    frames.append(img_bytes)
                else:
                    break
            cap.release()
            return frames

    def __call__(self, index_filename):
        index, filename = index_filename
        result = self.prepare(filename)
        return index, result


def prepare_data(data_type, path, out, n_worker, sizes, quality, chunksize):
    filenames = list()
    extensions = constants.IMAGE_EXTENSIONS if data_type == 'images' else constants.VIDEO_EXTENSIONS
    for ext in extensions:
        filenames += glob(f'{path}/**/*.{ext}', recursive=True)
    filenames = sorted(filenames)
    total = len(filenames)
    os.makedirs(out, exist_ok=True)

    for size in sizes:
        lmdb_path = os.path.join(out, str(size))
        with lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False) as env:
            with env.begin(write=True) as txn:
                txn.put(format_for_lmdb('length'), format_for_lmdb(total))
                resizer = Resizer(data_type, size=size, quality=quality)
                with multiprocessing.Pool(n_worker) as pool:
                    for idx, result in tqdm(
                            pool.imap_unordered(resizer, enumerate(filenames), chunksize=chunksize),
                            total=total):
                        if data_type == 'images':
                            txn.put(format_for_lmdb(idx), result)
                        else:
                            txn.put(format_for_lmdb(idx, 'length'), format_for_lmdb(len(result)))
                            for frame_idx, frame in enumerate(result):
                                txn.put(format_for_lmdb(idx, frame_idx), frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_type', type=str, help='data type', choices=['images', 'videos'])
    parser.add_argument('path', type=str, help='a path to input directiory')
    parser.add_argument('--out', type=str, help='a path to output directory')
    parser.add_argument('--sizes', type=int, nargs='+', default=(8, 16, 32, 64, 128, 256))
    parser.add_argument('--quality', type=int, help='output jpeg quality', default=85)
    parser.add_argument('--n_worker', type=int, help='number of worker processes', default=8)
    parser.add_argument('--chunksize', type=int, help='approximate chunksize for each worker', default=10)
    args = parser.parse_args()
    prepare_data(**vars(args))
