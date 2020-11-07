import random


def random_crop(tensor, size):
    assert tensor.dim() == 4, tensor.shape  # frames x chnnels x h x w
    h, w = tensor.shape[-2:]
    h_start = random.randint(0, h - size)
    w_start = random.randint(0, w - size)
    return tensor[:, :, h_start : h_start + size, w_start : w_start + size]


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return random_crop(tensor, self.size)


def random_horizontal_flip(tensor):
    flip = random.randint(0, 1)
    if flip:
        return tensor.flip(-1)
    else:
        return tensor


def identity(tensor):
    return tensor
