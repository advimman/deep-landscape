#!/usr/bin/env python3

import enum
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import constants
from inference.datasets import PrecomputedLatentDataset
from inference.datasets import move_to_device
from inference.encoder_train_pipeline import train_eval_loop
from inference.encoders import ResNetEncoder
from inference.inference_utils import load_generator_for_inference, get_noise_for_infer
from model import AdaptiveInstanceNorm, frames2batch


class EncoderMode(enum.Enum):
    LATENT_W_PRIME = 'latent_w_prime'
    LATENT_W_PRIME_NOISE = 'latent_w_prime_noise'


class DecoderFromLatents:
    def __init__(self, model_name, iteration, mode=EncoderMode.LATENT_W_PRIME_NOISE, generator_device='cuda:0'):
        self.mode = EncoderMode(mode)
        self.model_name = model_name
        self.iteration = iteration
        self.infer_model, config = load_generator_for_inference(self.model_name, self.iteration)
        self.generator_device = torch.device(generator_device)
        self.infer_model.g_running.to(self.generator_device)

    def clone(self):
        return DecoderFromLatents(self.model_name,
                                  self.iteration,
                                  mode=self.mode,
                                  generator_device=self.generator_device)

    def check_latents(self, latent_params):
        prefixes = ['latent_wprime']
        if self.mode == EncoderMode.LATENT_W_PRIME_NOISE:
            prefixes.append('noise')

        for i in range(self.infer_model.step + 1):
            for j in (0, 1):
                for prefix in prefixes:
                    key = f'{prefix}:{i}:{j}'
                    assert key in latent_params, f'{key} is missing in latents!'

    def __call__(self, latent_params):
        self.check_latents(latent_params)

        latent_params = move_to_device(latent_params, self.generator_device)
        some_w = latent_params['latent_wprime:0:0']
        batch_size = some_w.shape[0]

        if self.mode == EncoderMode.LATENT_W_PRIME_NOISE:
            noise = [(latent_params[f'noise:{i}:0'], latent_params[f'noise:{i}:1'])
                     for i in range(self.infer_model['step'] + 1)]
        else:
            noise = get_noise_for_infer(self.infer_model.g_running, batch_size, step=self.infer_model.step,
                                        device=self.generator_device)
        noise = move_to_device(noise, self.generator_device)

        for level_i, block in enumerate(self.infer_model['g_running'].generator.progression):
            if level_i > self.infer_model.step:
                break

            adain1_style = latent_params[f'latent_wprime:{level_i}:0']
            if adain1_style is not None:
                if adain1_style.dim() == 3:
                    adain1_style = frames2batch(adain1_style)
                block.adain1.fixed_style = [adain1_style.contiguous()]

            adain2_style = latent_params[f'latent_wprime:{level_i}:1']
            if adain2_style is not None:
                if adain2_style.dim() == 3:
                    adain2_style = frames2batch(adain2_style)
                block.adain2.fixed_style = [adain2_style.contiguous()]

        result = self.infer_model.g_running([some_w],
                                            latent_type='w',
                                            step=self.infer_model.step,
                                            alpha=self.infer_model.alpha,
                                            noise=noise)

        for module in self.infer_model.g_running.modules():
            if isinstance(module, AdaptiveInstanceNorm):
                module.fixed_style = None

        return result


class EncoderLoss:
    def __init__(self, **decoder_kwargs):
        self.base_loss = nn.L1Loss()
        self.decoder = DecoderFromLatents(mode=EncoderMode.LATENT_W_PRIME, **decoder_kwargs)

    def __call__(self, pred, target):
        metrics = {}
        result = 0

        orig = target['images']
        reconstructed = self.decoder(pred)
        if reconstructed.shape[-1] > orig.shape[-1]:
            reconstructed = F.interpolate(reconstructed.squeeze(1), size=orig.shape[-2:],
                                          mode='bicubic', align_corners=True).unsqueeze(1)

        joint = torch.stack([orig, reconstructed], dim=0)
        image_for_vis = joint.view(-1, *joint.shape[-3:])

        for key, cur_pred in pred.items():
            cur_target = target[key]
            cur_base = self.base_loss(cur_pred, cur_target)
            result = result + cur_base
            metrics['base/{}'.format(key)] = float(cur_base)

        return result, metrics, image_for_vis


def main(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]

    out_dir = os.path.join(constants.ENCODER_TRAIN_DIR, config_name)
    images_dir = os.path.join(out_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    all_latent_files = list(glob.glob(os.path.join(constants.RESULT_DIR, config['data'], '*.pth')))
    np.random.shuffle(all_latent_files)
    latent_train_split = int(len(all_latent_files) * config['train_size'])
    latent_train_files = all_latent_files[:latent_train_split]
    latent_val_files = all_latent_files[latent_train_split:]

    train_dataset = PrecomputedLatentDataset(latent_train_files, **config.get('dataset_kwargs', {}))
    val_dataset = PrecomputedLatentDataset(latent_val_files, **config.get('dataset_kwargs', {}))

    criterion = EncoderLoss(**config['loss_kwargs'])

    def lr_scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config['lr_scheduler'])

    model = ResNetEncoder(**config['encoder_kwargs'])

    (best_val_loss,
     best_metrics,
     best_model) = train_eval_loop(model, train_dataset, val_dataset, criterion,
                                   data_loader_ctor=lambda x, *args, **kwargs: x,
                                   lr_scheduler_ctor=lr_scheduler,
                                   save_vis_images_path=images_dir,
                                   save_models_path=models_dir,
                                   **config.get('train_eval_loop_kwargs', {}))


if __name__ == '__main__':
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument('config_path')

    main(aparser.parse_args())
