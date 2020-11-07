import numbers

import numpy as np
import torch
from torch.utils.data import IterableDataset


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [move_to_device(t, device) for t in obj]
    elif isinstance(obj, dict):
        return {name: move_to_device(t, device) for name, t in obj.items()}
    else:
        raise ValueError('Unexpected object type {}'.format(type(obj)))


def expand_latents(obj, name_prefix=''):
    result = {}
    if torch.is_tensor(obj):
        result[name_prefix] = obj
    elif isinstance(obj, (list, tuple)):
        for i, subobj in enumerate(obj):
            result.update(expand_latents(subobj, name_prefix=f'{name_prefix}:{i}'))
    elif isinstance(obj, dict):
        for name, subobj in obj.items():
            result.update(expand_latents(subobj, name_prefix=f'{name_prefix}{name}'))
    elif isinstance(obj, numbers.Number):
        result[name_prefix] = torch.tensor(obj).unsqueeze(0)
    else:
        raise ValueError('Unexpected object type {}'.format(type(obj)))
    return result


def get_shape(t):
    if torch.is_tensor(t):
        return t.shape
    elif isinstance(t, dict):
        return {n: get_shape(q) for n, q in t.items()}
    elif isinstance(t, (list, tuple)):
        return [get_shape(q) for q in t]
    elif isinstance(t, numbers.Number):
        return type(t)
    else:
        raise ValueError('unexpected type {}'.format(type(t)))


class PrecomputedLatentDataset(IterableDataset):
    def __init__(self, batch_files, latent_names=None, batch_size=None, mixup=None):
        self.batch_files = batch_files
        self.latent_names = latent_names
        self.batch_size = batch_size
        self.mixup = mixup

    def __iter__(self):
        prev_batch = None

        for batch_i in np.random.randint(0, len(self.batch_files), len(self.batch_files)):
            new_batch = torch.load(self.batch_files[batch_i])  #, map_location='cpu')

            if self.latent_names:
                new_batch = {name: new_batch[name] for name in self.latent_names}

            new_batch = expand_latents(new_batch)
            batch_size = next(iter(new_batch.values())).shape[0]
            if self.batch_size is not None:
                with torch.no_grad():
                    batch_idx = torch.randint(batch_size, size=(self.batch_size,))
                    new_batch = {name: value[batch_idx] for name, value in new_batch.items()}
                    batch_size = self.batch_size

            if prev_batch is not None:
                if self.mixup is None:
                    mask = torch.from_numpy(np.random.binomial(1, 0.5, size=batch_size))
                else:
                    mask = torch.from_numpy(np.random.beta(self.mixup, self.mixup, size=batch_size))

                res_batch = {}
                for name, new_value in new_batch.items():
                    mask = mask.to(new_value).view(batch_size, *((1,) * (new_value.ndim - 1)))
                    res_batch[name] = prev_batch[name] * mask + new_value * (1 - mask)
            else:
                res_batch = dict(new_batch)

            yield res_batch['images'], dict(res_batch)

            prev_batch = new_batch
