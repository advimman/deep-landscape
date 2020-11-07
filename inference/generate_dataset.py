#!/usr/bin/env python3

import os
import random

import torch
import torch.nn.functional as F
import tqdm

import constants
from inference.inference_utils import get_wprime, load_generator_for_inference, get_noise_for_infer
from model import StyleChangeMode
from utils import get_latents, get_mean_style


def scale_tensors(tensors, scale=1):
    if torch.is_tensor(tensors):
        return tensors * scale
    elif isinstance(tensors, (list, tuple)):
        return [scale_tensors(t, scale) for t in tensors]
    else:
        raise ValueError('Unexpected value type {}'.format(type(tensors)))


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device('cuda')

    infer_model, config = load_generator_for_inference(args.model_name, args.model_iter)

    code_size = config.get('code_size', constants.DEFAULT_CODE_SIZE)
    step = infer_model['step']

    style_change_mode = StyleChangeMode.RESAMPLE
    generator = infer_model['g_running'].to(device)
    mean_style = get_mean_style(generator, device, code_size)

    for batch_i in tqdm.trange(args.num_samples // args.batch_size):
        latent_z = get_latents(args.batch_size, code_size) * args.z_scale
        latent_w = generator.get_styles(latents=latent_z, mean_style=mean_style, style_weight=args.trunc,
                                        n_frames=1, change_mode=style_change_mode)

        if random.random() < args.mixin_prob:
            latent_z2 = get_latents(args.batch_size, code_size) * args.z_scale
            latent_w2 = generator.get_styles(latents=latent_z2, mean_style=mean_style, style_weight=args.trunc,
                                             n_frames=1, change_mode=style_change_mode)
            mixing_range = (random.randint(1, step), step + 1)
            latent_w = [latent_w[0], latent_w2[0]]
        else:
            mixing_range = (-1, -1)

        latent_wprime = get_wprime(generator, latent_w, mixing_range=mixing_range, max_step=step)

        noise = get_noise_for_infer(generator, args.batch_size, step=step, device=device)

        images = generator(latent_w, latent_type='w',
                           step=infer_model['step'],
                           alpha=infer_model['alpha'],
                           noise=noise,
                           mixing_range=mixing_range)

        if args.scale_images is not None and images.shape[-1] > args.scale_images:
            images = F.interpolate(images.squeeze(1), size=(args.scale_images, args.scale_images),
                                   mode='bicubic', align_corners=True).unsqueeze(1)

        batch = dict(latent_z=latent_z,
                     latent_w=latent_w,
                     latent_wprime=latent_wprime,
                     # noise=noise,
                     images=images,
                     mixing_range=mixing_range)
        torch.save(batch, os.path.join(args.outdir, '{:06d}.pth'.format(batch_i)))


if __name__ == '__main__':
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument('model_name')
    aparser.add_argument('outdir')
    aparser.add_argument('--num-samples', type=int, default=200000)
    aparser.add_argument('--z-scale', type=float, default=3)
    aparser.add_argument('--model-iter', type=int)
    aparser.add_argument('--batch-size', type=int, default=6)
    aparser.add_argument('--trunc', type=float, default=1)
    aparser.add_argument('--mixin-prob', type=float, default=0.9)
    aparser.add_argument('--scale-images', type=int, default=None)
    aparser.add_argument('--device', type=str, default='cuda')

    main(aparser.parse_args())
