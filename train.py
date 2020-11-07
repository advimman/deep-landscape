#!/usr/bin/env python3

import os
import argparse
import random
import math
import logging

import yaml
import tqdm
import torch
from torch.nn import functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

from datasets import MultiResolutionImageDataset, MultiResolutionMultiFrameDataset,\
    MultiResolutionMultiCropDataset
from utils import save_sample, get_model, accumulate, get_latents
import constants
from logger import setup_logger, LOGGER
import tensor_transforms


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class CycleLoader:
    def _init_loader(self):
        self.loader = iter(DataLoader(self.dataset, shuffle=True,
                                      batch_size=self.batch_size,
                                      num_workers=constants.NUM_WORKERS))

    def __init__(self, dataset, batch_size, resolution):
        dataset.resolution = resolution
        self.dataset = dataset
        self.batch_size = batch_size
        self._init_loader()

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self._init_loader()
            return next(self)


def discr_backward_real(discriminator, loss_fn, real_image, step, alpha):
    if loss_fn == 'wgan-gp':
        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        (-real_predict).backward()
        grad_loss_val = None

    elif loss_fn == 'r1':
        real_image.requires_grad = True
        LOGGER.debug(f'real image shape {real_image.shape}')
        real_scores = discriminator(real_image, step=step, alpha=alpha)
        real_predict = F.softplus(-real_scores).mean()
        real_predict.backward(retain_graph=True)

        grad_real = grad(
            outputs=real_scores.sum(), inputs=real_image, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()

    return real_predict, grad_loss_val


def discr_backward_fake(discriminator, loss_fn, fake_image, real_image, real_predict, step, alpha, is_n_frames_discr):
    fake_predict = discriminator(fake_image, step=step, alpha=alpha)
    if loss_fn == 'wgan-gp':
        fake_predict = fake_predict.mean()
        fake_predict.backward()

        eps = torch.rand(fake_image.shape[0], 1 if is_n_frames_discr else fake_image.shape[1], 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_penalty = (
            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()
        discr_loss_val = (real_predict - fake_predict).item()

    elif loss_fn == 'r1':
        fake_predict = F.softplus(fake_predict).mean()
        fake_predict.backward()
        discr_loss_val = (real_predict + fake_predict).item()
        grad_loss_val = None

    return discr_loss_val, grad_loss_val


def get_model_state(model):
    LOGGER.debug(f'Used samples on saving: {model.used_samples}.')
    state = dict()
    for key, value in model.items():
        if isinstance(value, (torch.nn.Module, torch.optim.Optimizer)):
            if isinstance(value, torch.nn.DataParallel):
                value = value.module
            state[key] = value.state_dict()
        elif isinstance(value, (float, int)):
            state[key] = value
        else:
            LOGGER.error(f'Model contains value {key} of wrong type {type(value)}')
            raise TypeError
    return state


class Trainer:
    def __init__(self, config_path, img_data_path, video_data_path, restart, from_step=False, debug=False):
        assert not restart or not from_step

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.model_name = os.path.basename(config_path)[:-len('.yaml')]
        self.setup_dirs()
        self.setup_loggers(debug)
        self.model = get_model(self.model_name, self.config, restart=restart, from_step=from_step)
        self.setup_datasets(img_data_path, video_data_path)

    def setup_loggers(self, debug):
        level = logging.DEBUG if debug else logging.INFO
        setup_logger(out_file=os.path.join(constants.LOG_DIR, 'train_' + self.model_name + '.log'),
                     stdout_level=level,
                     file_level=level)
        self.summary_writer = SummaryWriter(log_dir=os.path.join(constants.TB_DIR, 'train', self.model_name))

    def setup_dirs(self):
        os.makedirs(constants.LOG_DIR, exist_ok=True)
        self.checkpoint_dir = os.path.join(constants.CHECKPOINT_DIR, self.model_name)
        self.sample_dir = os.path.join(constants.SAMPLE_DIR, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

    def setup_datasets(self, img_data_path, video_data_path):
        # Setup datasets
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.images_dataset = MultiResolutionImageDataset(
            img_data_path,
            transforms.Compose([transforms.RandomHorizontalFlip(), transform]),
        )

        self.n_frames_dataset = None
        self.n_crops_dataset = None
        n_frames_params = self.config.get('n_frames_params', dict())
        n_frames = n_frames_params.get('n', 1)
        if n_frames > 1 and (video_data_path is None):
            raise ValueError(f'Need video data to train {n_frames}-frames model.')
        elif n_frames == 1 and (video_data_path is not None):
            raise ValueError("Cannot use video data to train 1-frame model.")
        if n_frames > 1:
            video_dataset_args = [video_data_path, transform, tensor_transforms.random_horizontal_flip, n_frames]
            self.n_frames_dataset = MultiResolutionMultiFrameDataset(*video_dataset_args)
            if n_frames_params.get('crop_freq', 0) > 0:
                self.n_crops_dataset = MultiResolutionMultiCropDataset(*video_dataset_args)

    def save_model(self, *, iteration=None, step=None):
        assert (iteration is None) != (step is None)
        if iteration is not None:
            LOGGER.info(f'Saving model on iteration {iteration}')
            save_name = str(iteration).zfill(6)
        else:
            LOGGER.info(f'Saving model on step {step}')
            if self.model.used_samples != 0:
                raise Exception
            save_name = f'train_step-{step}'
        torch.save(get_model_state(self.model), os.path.join(self.checkpoint_dir, f'{save_name}.model'))

    def adjust_lr(self, lr, resolution):
        for key in ('g_optimizer', 'd_optimizer', 'nfd_optimizer'):
            optimizer = self.model[key]
            for group in optimizer.param_groups:
                mult = group.get('mult', 1)
                group['lr'] = lr[resolution] * mult

    def run(self):
        try:
            # setting variables and constants
            model = self.model
            generator = model.generator.train()
            g_running = model.g_running
            discriminator = model.discriminator
            n_frames_discriminator = model.n_frames_discriminator
            g_optimizer = model.g_optimizer
            d_optimizer = model.d_optimizer
            nfd_optimizer = model.nfd_optimizer
            used_samples = model.used_samples
            step = model.step
            resolution = model.resolution
            iteration = model.iteration

            n_critic = constants.N_CRITIC

            config = self.config
            code_size = config.get('code_size', constants.DEFAULT_CODE_SIZE)
            lr = config.get('lr', constants.LR)
            batch_size = config.get('batch_size', constants.BATCH_SIZE)
            init_size = config.get('init_size', constants.INIT_SIZE)
            n_gen_steps = config.get('n_gen_steps', 1)
            max_size = config['max_size']
            max_iterations = config.get('max_iterations', constants.MAX_ITERATIONS)
            samples_per_phase = config['samples_per_phase']
            loss_fn = config['loss_fn']

            n_frames_params = config.get('n_frames_params', dict())
            n_frames = n_frames_params.get('n', 1)
            n_frames_loss_coef = n_frames_params.get('loss_coef', 0)
            n_frames_final_freq = n_frames_params.get('final_freq', 0)
            n_frames_decay_duration = n_frames_params.get('decay_duration', 0)
            crop_freq = n_frames_params.get('crop_freq', 0)
            mixing = config.get('mixing', False)

            # getting data
            cur_batch_size = batch_size[resolution]
            images_dataloader = CycleLoader(
                self.images_dataset, cur_batch_size, resolution
            )

            if n_frames_loss_coef > 0:
                n_frames_dataloader = CycleLoader(
                    self.n_frames_dataset, cur_batch_size, resolution
                )
                if crop_freq > 0:
                    n_crops_dataloader = CycleLoader(
                        self.n_crops_dataset, cur_batch_size, resolution
                    )

            if iteration == 0:
                self.adjust_lr(lr, resolution)

            pbar = tqdm.trange(iteration, max_iterations, initial=iteration)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

            discr_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            max_step = int(math.log2(max_size)) - 2
            final_progress = False

            for iteration in pbar:
                model.iteration = iteration

                # update alpha, step and resolution
                alpha = min(1, 1 / samples_per_phase * (used_samples + 1))
                if resolution == init_size or final_progress:
                    alpha = 1
                if not final_progress and used_samples > samples_per_phase * 2:
                    LOGGER.debug(f'Used samples: {used_samples}.')
                    used_samples = 0
                    step += 1
                    if step > max_step:
                        step = max_step
                        final_progress = True
                        LOGGER.info('Final progress.')
                    else:
                        alpha = 0
                        LOGGER.info(f'Changing resolution from {resolution} to {resolution * 2}.')
                    resolution = 4 * 2 ** step
                    model.step = step
                    model.resolution = resolution
                    model.used_samples = used_samples
                    LOGGER.debug(f'Used samples on saving: {model.used_samples}.')
                    self.save_model(step=step)
                    self.adjust_lr(lr, resolution)

                    # setup loaderts
                    cur_batch_size = batch_size[resolution]
                    images_dataloader = CycleLoader(
                        self.images_dataset, cur_batch_size, resolution
                    )
                    if n_frames_loss_coef > 0:
                        n_frames_dataloader = CycleLoader(
                            self.n_frames_dataset, cur_batch_size, resolution
                        )
                        if crop_freq > 0:
                            n_crops_dataloader = CycleLoader(
                                self.n_crops_dataset, cur_batch_size, resolution
                            )

                # decide if need to use n_frames on this iteration
                if final_progress or n_frames_decay_duration == 0:
                    n_frames_freq = n_frames_final_freq
                else:
                    n_frames_freq = 0.5 - min(1, used_samples / n_frames_decay_duration) *\
                        (0.5 - n_frames_final_freq)
                n_frames_iteration = True if random.random() < n_frames_freq else False
                if n_frames_iteration:
                    cur_discr = n_frames_discriminator
                    cur_dataloader = n_frames_dataloader
                    cur_n_frames = n_frames
                    cur_d_optimizer = nfd_optimizer
                else:
                    cur_discr = discriminator
                    cur_dataloader = images_dataloader
                    cur_n_frames = 1
                    cur_d_optimizer = d_optimizer

                cur_discr.zero_grad()
                real_image = next(cur_dataloader)
                LOGGER.debug(f'n_frames iteration: {n_frames_iteration}')
                LOGGER.debug(f'cur_discr: {type(cur_discr.module)}')
                LOGGER.debug(f'real_image shape {real_image.shape}; resolution {resolution}')

                # discriminator step
                real_predict, real_grad_loss_val = discr_backward_real(cur_discr, loss_fn, real_image, step, alpha)
                if mixing and random.random() < 0.9:
                    num_latents = 2
                else:
                    num_latents = 1
                LOGGER.debug(f'Batch size: {cur_batch_size}')
                latents = get_latents(cur_batch_size, code_size, 2 * num_latents)
                gen_in1 = latents[:num_latents]
                gen_in2 = latents[num_latents:]
                LOGGER.debug(f'Latents shape: {gen_in1[0].shape}')
                fake_image = generator(gen_in1, step=step, alpha=alpha, n_frames=cur_n_frames)

                crop_iteration = False
                if n_frames_iteration:
                    if random.random() < crop_freq:
                        crop_iteration = True
                        fake_image = next(n_crops_dataloader)
                discr_loss_val, fake_grad_loss_val = discr_backward_fake(
                    cur_discr, loss_fn, fake_image, real_image, real_predict, step, alpha, False)
                grad_loss_val = real_grad_loss_val or fake_grad_loss_val
                cur_d_optimizer.step()

                # generator step
                if (iteration + 1) % n_critic == 0:
                    for gen_step in range(n_gen_steps):
                        generator.zero_grad()

                        requires_grad(generator, True)
                        requires_grad(cur_discr, False)

                        fake_image = generator(gen_in2, step=step, alpha=alpha, n_frames=cur_n_frames)
                        LOGGER.debug(f'fake image shape when gen {fake_image.shape}')

                        predict = cur_discr(fake_image, step=step, alpha=alpha)
                        if loss_fn == 'wgan-gp':
                            loss = -predict.mean()
                        elif loss_fn == 'r1':
                            loss = F.softplus(-predict).mean()

                        if n_frames_iteration:
                            loss *= n_frames_loss_coef
                        gen_loss_val = loss.item()

                        loss.backward()
                        g_optimizer.step()
                        LOGGER.debug('generator optimizer step')
                        accumulate(to_model=g_running, from_model=generator.module)

                        requires_grad(generator, False)
                        requires_grad(cur_discr, True)

                used_samples += real_image.shape[0]
                model.used_samples = used_samples

                if (iteration + 1) % constants.SAMPLE_FREQUENCY == 0:
                    LOGGER.info(f'Saving samples on {iteration + 1} iteration.')
                    save_sample(generator=g_running, alpha=alpha, step=step, code_size=code_size,
                                resolution=resolution,
                                save_dir=os.path.join(self.sample_dir),
                                name=f'{str(iteration + 1).zfill(6)}',
                                sample_size=constants.SAMPLE_SIZE,
                                images_n_frames=n_frames, video_n_frames=32)

                if (iteration + 1) % constants.SAVE_FREQUENCY == 0:
                    self.save_model(iteration=iteration+1)

                if n_frames_iteration:
                    prefix = 'NF'
                    suffix = 'n_frames'
                else:
                    prefix = ''
                    suffix = 'loss'

                state_msg = f'Size: {resolution}; {prefix}G: {gen_loss_val:.3f}; {prefix}D: {discr_loss_val:.3f}; ' +\
                            f'{prefix}Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
                pbar.set_description(state_msg)

                if iteration % constants.LOG_LOSS_FREQUENCY == 0:
                    self.summary_writer.add_scalar('size', resolution, iteration)
                    self.summary_writer.add_scalar(f'G/{suffix}', gen_loss_val, iteration)
                    self.summary_writer.add_scalar(f'D/{suffix}', discr_loss_val, iteration)
                    self.summary_writer.add_scalar(f'Grad/{suffix}', grad_loss_val, iteration)
                    self.summary_writer.add_scalar('alpha', alpha, iteration)
                    if n_frames_iteration and crop_freq > 0:
                        if crop_iteration:
                            suffix = 'crop'
                        else:
                            suffix = 'no_crop'
                        self.summary_writer.add_scalar(f'D/{suffix}', discr_loss_val, iteration)

        except KeyboardInterrupt:
            LOGGER.warning('Interrupted by user')
            self.save_model(iteration=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('-i', '--img_data_path', required=True)
    parser.add_argument('-v', '--video_data_path', default=None)
    start_params_group = parser.add_mutually_exclusive_group()
    start_params_group.add_argument('--restart', action='store_true', help='Whether to restart training.')
    start_params_group.add_argument('--from_step', action='store_true',
                                    help='Whether to restart from step checkpoints.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    trainer = Trainer(**vars(args))
    trainer.run()
