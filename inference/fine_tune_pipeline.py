import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from PIL import Image
from torch.utils.data import Dataset

from inference.inference_utils import get_tqdm
from inference.perceptual_loss import PerceptualLoss
from model import frames2batch


def optimize_latents(images_batch, latents, decoder, device='cuda',
                     lr=1e-1, max_iters=500, early_stopping_patience=60, early_stopping_eps=1e-3,
                     reduce_lr_patience=20, reduce_lr_factor=0.5,
                     debug_frequency=None, debug_folder=None, return_metrics=False,
                     still_segm_mask=None, optimize_only_keys=None,
                     reconstr_l1_coef=1, pl_coef=0.01, w_init_l2_reg_coef=0,
                     noise_grad_scale=1,
                     latent_w_grad_scale_still=1, latent_w_grad_scale_movable=0.01,
                     latent_noise1_grad_scale_still=1, latent_noise1_grad_scale_movable=0,
                     latent_noise2_grad_scale_still=0, latent_noise2_grad_scale_movable=1,
                     reduce_lr_verbose=False
                     ):
    device = torch.device(device)

    latent_variables = {name: nn.Parameter(tensor.clone().detach().to(device))
                        for name, tensor in latents.items()}
    latent_variables_init_values = copy.deepcopy(latent_variables)

    if optimize_only_keys is None:
        optimize_only_keys = list(latents.keys())

    optimizer = torch.optim.Adam([latent_variables[name] for name in optimize_only_keys], lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              patience=reduce_lr_patience,
                                                              factor=reduce_lr_factor,
                                                              verbose=reduce_lr_verbose)

    gold_images = images_batch.to(device)

    l1_criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss().to(device)

    best_loss = float('inf')
    best_iter = 0

    all_metrics = []

    with torch.enable_grad():
        progress_bar = get_tqdm(range(max_iters), total=max_iters, desc='Iter', leave=True)
        for iter_i in progress_bar:
            metrics = {}

            use_masking = still_segm_mask is not None
            if use_masking:
                iter_for_still = iter_i % 2 == 1
                cur_mask = still_segm_mask if iter_for_still else (1 - still_segm_mask)

            pred_images = frames2batch(decoder(latent_variables))

            w_init_reg_value = 0
            if w_init_l2_reg_coef > 0:
                for name, var in latent_variables.items():
                    if name.startswith('latent_w'):
                        w_init_reg_value = w_init_reg_value + F.mse_loss(var,
                                                                         latent_variables_init_values[name].detach())
            w_init_reg_value = w_init_reg_value * w_init_l2_reg_coef

            pred_images_for_loss = (pred_images * cur_mask) if use_masking else pred_images
            gold_images_for_loss = (gold_images * cur_mask) if use_masking else gold_images

            pl_loss = perceptual_criterion(pred_images_for_loss, gold_images_for_loss).mean() * pl_coef
            l1_loss = l1_criterion(pred_images_for_loss, gold_images_for_loss) * reconstr_l1_coef

            optimizer.zero_grad()

            loss = l1_loss + pl_loss + w_init_reg_value
            loss.backward()

            metrics.update(dict(
                w_init_l2_reg=float(w_init_reg_value),
                pl_loss=float(pl_loss),
                l1_loss=float(l1_loss),
            ))

            if use_masking:
                for name in optimize_only_keys:
                    var = latent_variables[name]
                    if iter_for_still:
                        if name.startswith('noise') and name.endswith(':0'):
                            var.grad[:] *= latent_noise1_grad_scale_still
                        elif name.startswith('noise') and name.endswith(':1'):
                            var.grad[:] *= latent_noise2_grad_scale_still
                        elif name.startswith('latent_w'):
                            var.grad[:] *= latent_w_grad_scale_still
                    else:
                        if name.startswith('noise') and name.endswith(':0'):
                            var.grad[:] *= latent_noise1_grad_scale_movable
                        elif name.startswith('noise') and name.endswith(':1'):
                            var.grad[:] *= latent_noise2_grad_scale_movable
                        elif name.startswith('latent_w'):
                            var.grad[:] *= latent_w_grad_scale_movable

            if debug_frequency is not None and iter_i % debug_frequency == 0:
                torch.save(dict(latents=latent_variables,
                                images=pred_images,
                                metrics=metrics),
                           os.path.join(debug_folder, f'{iter_i:05d}.pth'))
            all_metrics.append(metrics)

            for name in optimize_only_keys:
                if name.startswith('noise'):
                    latent_variables[name].grad[:] *= noise_grad_scale

            optimizer.step()

            main_loss = metrics['l1_loss'] + metrics['pl_loss']
            lr_scheduler.step(main_loss)

            progress_bar.set_description('L1: {:.3f}, PL: {:.3f}'.format(metrics['l1_loss'],
                                                                         metrics['pl_loss']))

            if main_loss < best_loss - early_stopping_eps:
                best_iter = iter_i
                best_loss = main_loss
            elif iter_i - best_iter > early_stopping_patience:
                break

    result = {name: tensor.data.clone().detach()
              for name, tensor in latent_variables.items()}
    if return_metrics:
        return result, all_metrics
    return result


def fine_tune_generator(latents, images, decoder, max_iters=200, lr=1e-3, optimizer_ctor=torch.optim.Adam,
                        optimizer_kwargs=None, l1_coef=1, pl_coef=0.1, still_segm_mask=None,
                        reduce_lr_patience=1000, reduce_lr_factor=0.5):
    decoder = decoder.clone()
    default_optimizer_kwargs = dict(lr=lr)
    if optimizer_kwargs:
        default_optimizer_kwargs.update(optimizer_kwargs)
    optimizer = optimizer_ctor(decoder.infer_model['g_running'].parameters(), **default_optimizer_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              patience=reduce_lr_patience,
                                                              factor=reduce_lr_factor)
    decoder.infer_model['g_running'].train()

    l1_criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss().to(decoder.generator_device)
    images = images.to(decoder.generator_device)

    metrics = []
    progress_bar = get_tqdm(range(max_iters), total=max_iters, desc='Iter', leave=True)
    for iter_i in progress_bar:
        optimizer.zero_grad()

        reconstr = decoder(latents)

        reconstr_for_loss = (reconstr * still_segm_mask) if still_segm_mask is not None else reconstr
        reconstr_for_loss = reconstr_for_loss.squeeze(1)
        gold_for_loss = (images * still_segm_mask) if still_segm_mask is not None else images

        l1_value = l1_criterion(reconstr_for_loss, gold_for_loss) * l1_coef
        pl_value = perceptual_criterion(reconstr_for_loss, gold_for_loss) * pl_coef

        loss = l1_value + pl_value
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)

        metrics.append(dict(l1=float(l1_value),
                            pl=float(pl_value),
                            loss=float(loss)))
        progress_bar.set_description('L1: {:.3f}, PL: {:.3f}'.format(float(l1_value), float(pl_value)))

    decoder.infer_model['g_running'].eval()
    return decoder, metrics
