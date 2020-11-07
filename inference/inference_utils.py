import os

import tqdm
import yaml

from model import NoiseChangeMode
from utils import get_model


def get_wprime(generator, styles, mixing_range=(-1, -1), max_step=None):
    assert len(styles) in (1, 2)

    result = []

    for step_i, block in enumerate(generator.generator.progression):
        if len(styles) == 1:
            cur_style = styles[0]
        else:
            cur_style = styles[1] if mixing_range[0] <= step_i <= mixing_range[1] else styles[0]

        result.append((cur_style, cur_style))

        if step_i == max_step:
            break

    return result


def in_ipynb():
    try:
        get_ipython()
        return True
    except NameError:
        return False


def get_tqdm(*args, **kwargs):
    ctor = tqdm.notebook.tqdm if in_ipynb() else tqdm.tqdm
    return ctor(*args, **kwargs)


def get_trange(*args, **kwargs):
    ctor = tqdm.notebook.trange if in_ipynb() else tqdm.trange
    return ctor(*args, **kwargs)


def load_generator_for_inference(model_name, model_iter):
    config_path = os.path.join(os.path.dirname(__file__), '../configs/train', f'{model_name}.yaml')
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model_name = os.path.splitext(os.path.basename(config_path))[0]

    infer_model = get_model(model_name, config, iteration=model_iter, load_discriminator=False)
    return infer_model, config


def get_noise_for_infer(generator, batch_size, step, device='cuda', scale=1.):
    noise_change_modes = [NoiseChangeMode.RESAMPLE for _ in range(step + 1)]
    result = generator.get_noise(batch_size, n_frames=1, step=step, device=device,
                                 noise_change_modes=noise_change_modes, inversed=False)
    result = [(n1 * scale, n2 * scale) for n1, n2 in result]
    return result


def sum_dicts(result, cur_dict, prefix=''):
    for n, v in cur_dict.items():
        result[prefix + n] += v
