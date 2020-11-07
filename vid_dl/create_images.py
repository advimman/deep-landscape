#!/usr/bin/env python3

import os
import glob
import argparse
from pathlib import Path
from copy import deepcopy

import imageio
import cv2
from tqdm import tqdm


def create_images(input_dir, output_dir, resolution):
    for f in tqdm(glob.glob(os.path.join(input_dir, "*.mp4"))):
        vid = imageio.get_reader(f, 'ffmpeg')
        folder = os.path.split(f)[-1].rsplit(".", maxsplit=1)[0]
        Path(os.path.join(output_dir, folder)).mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(vid.iter_data()):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            new_frame = deepcopy(frame)
            path = os.path.join(output_dir, folder, str(i).zfill(5) + ".jpg")
            if new_frame.shape[0] != resolution:
                new_frame = cv2.resize(new_frame, (resolution, resolution))
            cv2.imwrite(path, new_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--resolution', type=int, default=256)
    args = parser.parse_args()
    create_images(**vars(args))
