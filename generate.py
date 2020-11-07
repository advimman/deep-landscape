#!/usr/bin/env python3

import argparse
import os
import logging

import yaml

from utils import get_model, save_sample
from model import NoiseChangeMode, StyleChangeMode
import constants
from logger import setup_logger


def generate(config_path, iteration, trunc, debug, image_n_frames, video_n_frames, change_modes,
             inversed=False, homography_dir=None, separate_files=False, num_files=1, save_frames=False):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_name = os.path.basename(config_path)[:-len('.yaml')]

    os.makedirs(constants.LOG_DIR, exist_ok=True)
    setup_logger(out_file=os.path.join(constants.LOG_DIR, 'gen_' + model_name + '.log'),
                 stdout_level=logging.DEBUG if debug else logging.INFO)

    gen_model = get_model(model_name=model_name, config=config, iteration=iteration)
    gen_path = os.path.join(constants.GEN_DIR, model_name)
    os.makedirs(gen_path, exist_ok=True)

    generator = gen_model['g_running'].eval()
    code_size = config.get('code_size', constants.DEFAULT_CODE_SIZE)
    alpha = gen_model['alpha']
    step = gen_model['step']
    resolution = gen_model['resolution']
    iteration = gen_model['iteration']

    for mode in change_modes.split(','):
        assert mode in available_modes, mode
        if mode.startswith('noise'):
            style_change_mode = StyleChangeMode.REPEAT
        else:
            style_change_mode = StyleChangeMode.INTERPOLATE

        if mode == 'style':
            noise_change_mode = NoiseChangeMode.FIXED
        elif mode.endswith('homography'):
            noise_change_mode = NoiseChangeMode.HOMOGRAPHY
            assert homography_dir is not None, 'The homography mode needs a path to a homography directory!'
        else:
            noise_change_mode = NoiseChangeMode.SHIFT
        noise_change_modes = [noise_change_mode] * constants.MAX_LAYERS_NUM

        if mode == 'images':
            save_video = False
            save_images = True
        else:
            save_video = True
            save_images = False

        save_dir = os.path.join(gen_path, mode)
        if mode.endswith('homography'):
            save_dir = os.path.join(save_dir, os.path.basename(homography_dir))

        save_sample(generator, alpha, step, code_size, resolution,
                    save_dir=save_dir,
                    name=('inversed_' if inversed else '') + str(iteration+1).zfill(6),
                    sample_size=constants.SAMPLE_SIZE, truncation_psi=trunc,
                    images_n_frames=image_n_frames,
                    video_n_frames=video_n_frames,
                    save_images=save_images, save_video=save_video,
                    style_change_mode=style_change_mode, noise_change_modes=noise_change_modes,
                    inversed=inversed, homography_dir=homography_dir,
                    separate_files=separate_files, num_files=num_files, save_frames=save_frames)


if __name__ == '__main__':
    available_modes = set(('homography', 'noise_homography', 'shift', 'noise', 'style', 'images'))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_path')
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--image_n_frames', type=int, default=constants.IMAGE_N_FRAMES)
    parser.add_argument('--video_n_frames', type=int, default=constants.VIDEO_N_FRAMES)
    parser.add_argument('--change_modes',
                        help=f'should be subset of {available_modes} splitted by commas',
                        default='homography')
    parser.add_argument('--inversed', action='store_true')
    parser.add_argument('--homography_dir')
    parser.add_argument('--separate_files', action='store_true')
    parser.add_argument('--save_frames', action='store_true')
    parser.add_argument('--num_files', default=1, type=int)
    parser.add_argument('--trunc', type=float, default=constants.TRUNCATION_PSI)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    generate(**vars(args))
