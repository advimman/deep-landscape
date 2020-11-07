#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from easydict import EasyDict as edict
import constants
from inference.encoders import MLPApproximator
from inference.encoder_train_pipeline import train_eval_loop
from inference.inference_utils import load_generator_for_inference


class MLPDataset(Dataset):
    def __init__(self, mlp, style_size=512, resample_n=3, batch_size=64, length=1000, device='cuda'):
        self.mlp = mlp
        self.mlp.to(device)
        self.device = torch.device(device)
        self.length = length
        self.batch_size = batch_size
        self.style_size = style_size
        self.resample_n = resample_n

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        with torch.no_grad():
            dyn_z1 = torch.randn(self.batch_size, self.resample_n, device=self.device)
            dyn_z2_far_end = torch.randn(self.batch_size, self.resample_n, device=self.device)
            interp_distance = torch.rand(1, device=self.device)
            dyn_z2 = dyn_z1 * (1 - interp_distance).pow(0.5) + dyn_z2_far_end * interp_distance.pow(0.5)

            const_z = torch.randn(self.batch_size, self.style_size - self.resample_n, device=self.device)

            z1 = torch.cat((const_z, dyn_z1), dim=-1)
            z2 = torch.cat((const_z, dyn_z2), dim=-1)

            all_z = torch.cat((z1, z2), dim=0)
            all_w = self.mlp(all_z)
            w1, w2 = all_w.chunk(2, 0)
            return (torch.cat((w1, dyn_z2_far_end, interp_distance.expand(self.batch_size, 1)), dim=1),
                    dict(w1=w1, w2=w2))


class MLPApproximateLoss:
    def __init__(self, abs_l1_coef=1, rel_l1_coef=1, rel_cos_coef=1):
        self.abs_l1_coef = abs_l1_coef
        self.rel_l1_coef = rel_l1_coef
        self.rel_cos_coef = rel_cos_coef

    def __call__(self, pred_w2, target):
        abs_l1 = F.l1_loss(pred_w2, target['w2']) * self.abs_l1_coef

        pred_dw = pred_w2 - target['w1']
        target_dw = target['w2'] - target['w1']
        rel_l1 = F.l1_loss(pred_dw, target_dw) * self.rel_l1_coef
        rel_cos = (1 - F.cosine_similarity(pred_dw, target_dw, dim=-1)).mean() * self.rel_cos_coef

        total_loss = abs_l1 + rel_l1 + rel_cos

        metrics = dict(abs_l1=float(abs_l1),
                       rel_l1=float(rel_l1),
                       rel_cos=float(rel_cos))
        return total_loss, metrics, None


def main(args):
    with open(args.config_path) as f:
        config = edict(yaml.load(f))
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]

    out_dir = os.path.join(constants.ENCODER_TRAIN_DIR, config_name)
    models_dir = os.path.join(out_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    infer_model = load_generator_for_inference(**config.generator_kwargs)[0]

    model = MLPApproximator(**config.model_kwargs)
    dataset = MLPDataset(infer_model['g_running'].style, **config.dataset_kwargs)
    criterion = MLPApproximateLoss(**config.criterion_kwargs)

    def lr_scheduler(optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.lr_scheduler)

    (best_val_loss,
     best_metrics,
     best_model) = train_eval_loop(model, dataset, dataset, criterion,
                                   data_loader_ctor=lambda x, *args, **kwargs: x,
                                   lr_scheduler_ctor=lr_scheduler,
                                   save_models_path=models_dir,
                                   **config.get('train_eval_loop_kwargs', {}))


if __name__ == '__main__':
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument('config_path')

    main(aparser.parse_args())
