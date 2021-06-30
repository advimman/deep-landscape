import math
import os
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch import optim
from torchvision import utils
from easydict import EasyDict

from model import StyledGenerator, Discriminator, NFramesDiscriminator, \
    StyleChangeMode, NoiseChangeMode
import constants
from logger import LOGGER


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


def load_yaml(path: str) -> dict:
    with open(path, 'r') as fin:
        result = yaml.load(fin, Loader=yaml.FullLoader)
    return result


def get_latents(shape, code_size, num=None):
    if isinstance(shape, int):
        shape = [shape]

    latents = torch.randn(num or 1, *shape, code_size, device='cuda')
    if num is not None:
        return list(latents)
    else:
        return latents[0]


@torch.no_grad()
def get_mean_style(generator, device, code_size=512, batches_n=100):
    mean_style = None
    batch_size = 1024

    for i in range(batches_n):
        style = generator.mean_style(get_latents(batch_size, code_size))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style

    mean_style /= batches_n
    return mean_style


def get_sample_row_col(resolution, sample_size=constants.SAMPLE_SIZE):
    return (sample_size[0] // resolution, sample_size[1] // resolution)


def save_image_samples(save_dir, name, latents, nrow, ncol, generator, step, alpha,
                       mean_style, truncation_psi, n_frames, inversed=False, separate_files=False):
    assert not separate_files or ncol == 1, 'When save samples in separate_files ncol should be 1.'
    images = []
    for i in range(nrow):
        imgs = generator(latents[i], step=step, alpha=alpha, mean_style=mean_style,
                         style_weight=truncation_psi, n_frames=n_frames, inversed=inversed)
        imgs = imgs.data.cpu()
        if separate_files:
            utils.save_image(
                imgs,
                os.path.join(save_dir, f'{i}_{name}.jpg'),
                nrow=ncol,
                normalize=True,
                range=(-1, 1),
            )
        else:
            images.append(imgs)

    if not separate_files:
        images = torch.stack(images, dim=2).transpose(0, 2)
        images = images.reshape(images.shape[0] * images.shape[1] * images.shape[2], *images.shape[3:])
        utils.save_image(
            images,
            os.path.join(save_dir, name + '.jpg'),
            nrow=ncol,
            normalize=True,
            range=(-1, 1),
        )


def write_video(out_path, frames, fps=constants.FPS):
    """Save 4d tensor as mp4 file
    :param out_path: str, where to save the video
    :param frames: FramesN x Channels x Height x Width float tensor with values from range [0, 1]
    """
    frames = frames[:, [2, 1, 0]]
    frames = (frames * 255).byte().detach().cpu().numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))
    num_frames, height, width, channels_n = frames.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def get_homographies(homography_dir):
    if homography_dir is None:
        return None
    homographies = []
    for path in glob(os.path.join(homography_dir, '*.csv')):
        with open(path) as f:
            shifts = f.readlines()
        shifts = (s.strip().split() for s in shifts)
        shifts = [(float(s[0]), float(s[1])) for s in shifts]
        homographies.append([[shifts[0], shifts[1]], [shifts[2], shifts[3]]])
    return homographies


def save_video_samples(save_dir, name, latents, nrow, ncol,
                       generator, step, alpha, mean_style, truncation_psi, resolution,
                       style_change_mode, noise_change_modes,
                       n_frames, inversed=False, homography_dir=None,
                       separate_files=False, save_frames=False):
    assert not separate_files or ncol == 1, 'When save samples in separate_files ncol should be 1.'
    if save_frames:
        os.makedirs(os.path.join(save_dir, 'frames'), exist_ok=True)
    images = []
    for i in range(nrow):
        for j in range(ncol):
            image = generator(
                [z[[j]] for z in latents[i]],
                step=step,
                alpha=alpha,
                n_frames=n_frames,
                style_change_mode=style_change_mode,
                noise_change_modes=noise_change_modes,
                mean_style=mean_style,
                style_weight=truncation_psi,
                inversed=inversed,
                homographies=get_homographies(homography_dir),
            )
            if separate_files:
                video = []
                for i_f in range(n_frames):
                    frame = utils.make_grid(
                        image[0][i_f],
                        nrow=ncol,
                        normalize=True,
                        range=(-1, 1),
                    )
                    if save_frames:
                        utils.save_image(frame, os.path.join(save_dir, 'frames', f'{name}_{i_f}.jpg'), nrow=1)
                    video.append(frame)
                video = torch.stack(video)
                write_video(os.path.join(save_dir, name + '.mp4'), video)
            else:
                images.append(image.data.cpu())

    if not separate_files:
        images = torch.cat(images, dim=0)
        video = []
        for i_f in range(n_frames):
            frame = utils.make_grid(
                images[:, i_f],
                nrow=ncol,
                normalize=True,
                range=(-1, 1),
            )
            if save_frames:
                utils.save_image(frame, os.path.join(save_dir, 'frames', f'{name}_frame_{i_f}.jpg'), nrow=1)
            video.append(frame)
        video = torch.stack(video)
        write_video(os.path.join(save_dir, name + '.mp4'), video)


def save_sample(generator, alpha, step, code_size, resolution, save_dir, name,
                sample_size=constants.SAMPLE_SIZE, fixed_latents=True,
                truncation_psi=constants.TRUNCATION_PSI,
                images_n_frames=constants.IMAGE_N_FRAMES, video_n_frames=constants.VIDEO_N_FRAMES,
                mixing=True, save_images=True, save_video=True,
                style_change_mode=StyleChangeMode.INTERPOLATE,
                noise_change_modes=tuple([NoiseChangeMode.SHIFT]*constants.MAX_LAYERS_NUM),
                inversed=False, homography_dir=None,
                separate_files=False, num_files=1, save_frames=False):

    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    for i in range(num_files):
        if num_files > 1:
            save_name = name + f'_{i}'
        else:
            save_name = name
        with torch.no_grad():
            if truncation_psi is not None:
                mean_style = get_mean_style(generator, 'cuda', code_size)
            else:
                mean_style = None

            if separate_files:
                nrow = num_files
                ncol = 1
            else:
                nrow, ncol = get_sample_row_col(resolution, sample_size)
            latents = get_latents((nrow, ncol), code_size, constants.MIXING_NUM)
            if not mixing:
                latents = latents[:1]
            latents = list(zip(*latents))  # nrow tuples of ncol x code_size

            if save_images:
                img_nrow = max(nrow // images_n_frames, 1)
                save_image_samples(save_dir, save_name, latents[:img_nrow], 1, ncol,
                                   generator, step, alpha, mean_style, truncation_psi,
                                   images_n_frames, inversed, separate_files)

            if save_video:
                save_video_samples(save_dir, save_name, latents, 1, ncol,
                                   generator, step, alpha, mean_style,
                                   truncation_psi, resolution,
                                   style_change_mode, noise_change_modes,
                                   video_n_frames, inversed, homography_dir,
                                   separate_files, save_frames)


def get_last_model(model_name, from_step=False):
    LOGGER.debug(f'Model checkpoint directory: {os.path.join(constants.CHECKPOINT_DIR, model_name)}')
    if from_step:
        model_paths = glob(os.path.join(constants.CHECKPOINT_DIR, model_name) + '/train_step-*.model')
        LOGGER.debug(f'Model paths: {model_paths}')
        last_step = max([int(os.path.basename(mp)[len('train_step-'):-len('.model')]) for mp in model_paths])
        last_model_path = os.path.join(constants.CHECKPOINT_DIR, model_name, f'train_step-{last_step}.model')
    else:
        model_paths = glob(os.path.join(constants.CHECKPOINT_DIR, model_name) + '/[!train_step-]*.model')
        LOGGER.debug(f'Model paths: {model_paths}')
        last_iter = max([int(os.path.basename(mp)[: -len('.model')]) for mp in model_paths])
        last_model_path = os.path.join(constants.CHECKPOINT_DIR, model_name, f'{str(last_iter).zfill(6)}.model')
    LOGGER.info(f'Loading {last_model_path} (last)')
    model = torch.load(last_model_path)
    return model


def accumulate(*, from_model, to_model, decay=0.999):
    from_params = dict(from_model.named_parameters())
    to_params = dict(to_model.named_parameters())
    for k in from_params.keys():
        to_params[k].data.mul_(decay).add_(from_params[k].data, alpha=(1 - decay))


def get_model(model_name, config, iteration=None, restart=False, from_step=False, load_discriminator=True,
              alpha=1, step=6, resolution=256, used_samples=0):
    """
    Function that creates a model.
    Arguments:
        model_name -- name to use for save and load the model.
        config -- dict of model parameters.
        iteration -- iteration to load; last if None
        restart -- if true, than creates new model even there is a saved model with `model_name`.
    """
    LOGGER.info(f'Getting model "{model_name}"')
    code_size = config.get('code_size', constants.DEFAULT_CODE_SIZE)
    init_size = config.get('init_size', constants.INIT_SIZE)
    n_frames_params = config.get('n_frames_params', dict())
    n_frames = n_frames_params.get('n', 1)
    from_rgb_activate = config['from_rgb_activate']
    two_noises = n_frames_params.get('two_noises', False)
    lr = config.get('lr', constants.LR)
    dyn_style_coordinates = n_frames_params.get('dyn_style_coordinates', 0)

    generator = nn.DataParallel(StyledGenerator(code_size,
                                                two_noises=two_noises,
                                                dyn_style_coordinates=dyn_style_coordinates,
                                                )).cuda()
    g_running = StyledGenerator(code_size,
                                two_noises=two_noises,
                                dyn_style_coordinates=dyn_style_coordinates,
                                ).cuda()
    g_running.train(False)
    discriminator = nn.DataParallel(Discriminator(from_rgb_activate=from_rgb_activate)).cuda()
    n_frames_discriminator = nn.DataParallel(
        NFramesDiscriminator(from_rgb_activate=from_rgb_activate, n_frames=n_frames)
    ).cuda()

    if not restart:
        if iteration is None:
            model = get_last_model(model_name, from_step)
        else:
            iteration = str(iteration).zfill(6)
            checkpoint_path = os.path.join(constants.CHECKPOINT_DIR, model_name, f'{iteration}.model')
            LOGGER.info(f'Loading {checkpoint_path}')
            model = torch.load(checkpoint_path)
        generator.module.load_state_dict(model['generator'])
        g_running.load_state_dict(model['g_running'])
        if load_discriminator:
            discriminator.module.load_state_dict(model['discriminator'])
        if 'n_frames_params' in config:
            n_frames_discriminator.module.load_state_dict(model['n_frames_discriminator'])
        alpha = model['alpha']
        step = model['step']
        LOGGER.debug(f'Step: {step}')
        resolution = model['resolution']
        used_samples = model['used_samples']
        LOGGER.debug(f'Used samples: {used_samples}.')
        iteration = model['iteration']
    else:
        alpha = 0
        step = int(math.log2(init_size)) - 2
        resolution = 4 * 2 ** step
        used_samples = 0
        iteration = 0
        accumulate(to_model=g_running, from_model=generator.module, decay=0)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(),
        lr=lr[resolution], betas=(0.0, 0.99)
    )

    style_module = generator.module
    style_params = list(style_module.style.parameters())
    g_optimizer.add_param_group(
        {
            'params': style_params,
            'lr': lr[resolution] * 0.01,
            'mult': 0.01,
        }
    )

    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr[resolution], betas=(0.0, 0.99))
    nfd_optimizer = optim.Adam(n_frames_discriminator.parameters(), lr=lr[resolution], betas=(0.0, 0.99))

    if not restart:
        g_optimizer.load_state_dict(model['g_optimizer'])
        d_optimizer.load_state_dict(model['d_optimizer'])
        nfd_optimizer.load_state_dict(model['nfd_optimizer'])

    return EasyDict(
           generator=generator,
           discriminator=discriminator,
           n_frames_discriminator=n_frames_discriminator,
           g_running=g_running,
           g_optimizer=g_optimizer,
           d_optimizer=d_optimizer,
           nfd_optimizer=nfd_optimizer,
           alpha=alpha,
           step=step,
           resolution=resolution,
           used_samples=used_samples,
           iteration=iteration,
       )
