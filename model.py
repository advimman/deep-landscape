import random
from math import sqrt
from enum import Enum, auto

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import init

import constants
from logger import LOGGER
from inference.perspective import make_homography_kornia, warp_homography_kornia


def slerp(val, low, high):
    """
    val, low, high: bs x frames x coordinates
    if val == 0 then low
    if val == 1 then high
    """
    assert low.dim() == 3, low.dim()
    assert val.dim() == 3, val.dim()
    assert high.dim() == 3, high.dim()
    low_norm = low / torch.norm(low, dim=2, keepdim=True)
    high_norm = high / torch.norm(high, dim=2, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(dim=2)).unsqueeze(-1)  # bs x frames x 1
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res


def bilinear_warp(images, flow, cycle_wrap=False, padding_mode='zeros'):
    """
    Apply warping via bilinear resampling to given images
    :param images: BatchSize x Channels x Height x Width - Images to warp
    :param flow: BatchSize x 2 x Height x Width - Offsets in range (-1, 1) (flow)
    :param cycle_wrap: Whether to append fragments moved out of view to another part of the image
    :return:
    """
    flow = flow[:, [1, 0]]
    batch_size, channels, height, width = images.size()
    height_coords = torch.linspace(-1, 1, height, device=flow.device)
    width_coords = torch.linspace(-1, 1, width, device=flow.device)
    src_grid = torch.cat([width_coords.unsqueeze(0).expand(height, width).unsqueeze(0),
                          height_coords.unsqueeze(1).expand(height, width).unsqueeze(0)],
                         dim=0).unsqueeze(0)
    new_grids = src_grid + flow
    if cycle_wrap:
        new_grids = (new_grids <= 1).float() * new_grids + (new_grids > 1).float() * new_grids % 2
        new_grids = (new_grids >= -1).float() * new_grids + (new_grids < -1).float() * new_grids % -2
        new_grids = new_grids - 2 * (new_grids > 1).float() + 2 * (new_grids < -1).float()
    return F.grid_sample(images, new_grids.permute(0, 2, 3, 1), padding_mode=padding_mode)


def frames2batch(tensor_or_list):
    if isinstance(tensor_or_list, list):
        return [frames2batch(t) for t in tensor_or_list]
    if isinstance(tensor_or_list, tuple):
        return tuple([frames2batch(t) for t in tensor_or_list])
    else:
        t = tensor_or_list
        return t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])


def batch2frames(tensor_or_list, batch_size, n_frames):
    if isinstance(tensor_or_list, list):
        return [batch2frames(t, batch_size, n_frames) for t in tensor_or_list]
    elif isinstance(tensor_or_list, tuple):
        return tuple([batch2frames(t, batch_size, n_frames) for t in tensor_or_list])
    else:
        t = tensor_or_list
        return t.view(batch_size, n_frames, *t.shape[1:])


def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        weight = torch.randn(in_channel, out_channel, *kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size[0] * kernel_size[1]
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        weight = torch.randn(out_channel, in_channel, *kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size[0] * kernel_size[1]
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding2 if padding2 is not None else padding

        kernel1 = kernel_size
        kernel2 = kernel_size2 if kernel_size2 is not None else kernel_size

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

        # monkey patch to optimize W'
        self.fixed_style = None

    def forward(self, input, style):
        # monkey patch to optimize W'
        if self.fixed_style is not None:
            assert self.fixed_style[0].shape == style.shape, (self.fixed_style[0].shape, style.shape)
            style = self.fixed_style[0]

        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        added = self.weight * noise
        return image + added


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        if isinstance(input, (list, tuple)):
            batch = input[0].shape[0]
        else:
            batch = input.shape[0]

        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
        two_noises=False,
        frames_channels=None
    ):
        super().__init__()
        self.two_noises = two_noises

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        if self.two_noises:
            self.noise12 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        if self.two_noises:
            self.noise22 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        if self.two_noises:
            noise1, noise2 = noise
        else:
            noise1 = noise

        if isinstance(style, tuple):
            style1, style2 = style
        else:
            style1 = style2 = style

        out = self.conv1(input)
        out = self.noise1(out, noise1)
        if self.two_noises:
            out = self.noise12(out, noise2)
        out = self.lrelu1(out)
        out = self.adain1(out, style1)

        out = self.conv2(out)
        out = self.noise2(out, noise1)
        if self.two_noises:
            out = self.noise22(out, noise2)
        out = self.lrelu2(out)
        out = self.adain2(out, style2)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim=512, fused=True, two_noises=False):
        super().__init__()
        channels = 512
        progression = []
        to_rgb = []
        for i in range(9):
            if i == 0:  # 4
                block = StyledConvBlock(
                    channels, channels, 3, 1, style_dim=code_dim, initial=True, two_noises=two_noises,
                    )
            elif i <= 3:  # 8 - 32
                block = StyledConvBlock(
                    channels, channels, 3, 1, style_dim=code_dim, upsample=True, two_noises=two_noises,
                    )
            elif i == 4:  # 64
                block = StyledConvBlock(
                    channels, channels // 2, 3, 1, style_dim=code_dim,
                    upsample=True, two_noises=two_noises,
                    )
                channels //= 2
            else:  # 128 - 1024
                block = StyledConvBlock(
                    channels, channels // 2, 3, 1, style_dim=code_dim, upsample=True,
                    fused=fused, two_noises=two_noises,
                    )
                channels //= 2
            progression.append(block)
            to_rgb.append(EqualConv2d(channels, 3, 1))

        self.progression = nn.ModuleList(progression)

        self.to_rgb = nn.ModuleList(to_rgb)

        # self.blur = Blur()

    def forward(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]
        if isinstance(out, tuple):
            out = out[0]

        if len(style) < 2 or step == 0:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

                out = conv(out, style_step, noise[i])

            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyleChangeMode(Enum):
    REPEAT = auto()
    INTERPOLATE = auto()
    RESAMPLE = auto()


class NoiseChangeMode(Enum):
    RESAMPLE = auto()
    SHIFT = auto()
    FIXED = auto()
    HOMOGRAPHY = auto()


def gen_stationary_step_noise(batch_size, n_frames, size, device):
    stationary_noise = torch.randn(batch_size, 1, 1, size, size, device=device).repeat(1, n_frames, 1, 1, 1)
    return stationary_noise


def gen_dyn_step_noise(batch_size, n_frames, size, device, noise_change_mode,
                       shift_values, homographies, h_nums):
    if noise_change_mode == NoiseChangeMode.RESAMPLE:
        dyn_noise = torch.randn(batch_size, n_frames, 1, size, size, device=device)
    elif noise_change_mode == NoiseChangeMode.FIXED:
        dyn_noise = torch.randn(batch_size, 1, 1, size, size, device=device).repeat(1, n_frames, 1, 1, 1)
    elif noise_change_mode == NoiseChangeMode.SHIFT:
        big = torch.randn(batch_size, 1, size * 2, size * 2, device=device)
        dyn_noise = []
        for frame_shift_value in shift_values:
            flow = frame_shift_value[None, :, None, None].repeat(batch_size, 1, size * 2, size * 2)
            frame_noise = bilinear_warp(big, flow, cycle_wrap=True)[:, :, :size, :size]
            frame_noise = frame_noise[:, :, :size, :size]
            dyn_noise.append(frame_noise)
        dyn_noise = torch.stack(dyn_noise, dim=1)
    elif noise_change_mode == NoiseChangeMode.HOMOGRAPHY:
        first_frame = torch.randn(batch_size, 1, size, size, device=device)
        dyn_noise = [first_frame]
        for f_i in range(1, n_frames):
            frame_dyn_noise = []
            start = 0
            for hn, h in zip(h_nums, homographies):
                if hn > 0:
                    frame_dyn_noise.append(warp_homography_kornia(
                        first_frame[start:hn], h, n_iter=f_i, horizon_line=constants.HORIZON_LINE))
                    start += hn
            frame_dyn_noise = torch.cat(frame_dyn_noise, dim=0)
            dyn_noise.append(frame_dyn_noise)
        dyn_noise = torch.stack(dyn_noise, dim=1)
    return dyn_noise


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8, two_noises=False, dyn_style_coordinates=0):
        super().__init__()
        self.dyn_style_coordinates = dyn_style_coordinates
        self.two_noises = two_noises
        self.code_dim = code_dim
        self.generator = Generator(code_dim, two_noises=two_noises)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

    def gen_step_noise(self, batch_size, n_frames, size, device, noise_change_mode,
                       shift_values, inversed=False, homographies=None, h_nums=None, perm=None):
        if self.two_noises or inversed:
            stationary_noise = gen_stationary_step_noise(batch_size, n_frames, size, device)
        dyn_noise = gen_dyn_step_noise(batch_size, n_frames, size, device, noise_change_mode,
                                       shift_values, homographies, h_nums)

        if noise_change_mode == NoiseChangeMode.HOMOGRAPHY:
            dyn_noise = dyn_noise[perm]

        if inversed:
            stationary_noise, dyn_noise = dyn_noise, stationary_noise

        if self.two_noises:
            return (stationary_noise, dyn_noise)
        else:
            return dyn_noise

    def get_noise(self, batch_size, n_frames, step, device, noise_change_modes,
                  inversed=False, homographies=None):
        if NoiseChangeMode.SHIFT in noise_change_modes:
            velocity = torch.randn(2, device=device) * constants.VELOCITY
            shift_values = velocity.unsqueeze(0) * torch.linspace(-1., 1., n_frames, device=device).unsqueeze(1)
        else:
            shift_values = None

        if NoiseChangeMode.HOMOGRAPHY in noise_change_modes:
            prepared_homographies = []
            for h in homographies:
                prepared_homographies.append(make_homography_kornia(h, constants.HORIZON_LINE))

            h_indexes = random.choices(list(range(len(homographies))), k=batch_size)
            h_nums = [0 for h in homographies]
            for h_i in h_indexes:
                h_nums[h_i] += 1
            assert sum(h_nums) == batch_size, h_nums
            perm = torch.randperm(batch_size)
            homographies = prepared_homographies
        else:
            h_nums = None
            perm = None

        noise = []
        size = 4
        for i in range(step + 1):
            step_noise = self.gen_step_noise(batch_size, n_frames, size, device,
                                             noise_change_modes[i], shift_values,
                                             inversed, homographies, h_nums, perm)
            noise.append(step_noise)
            size *= 2
        return noise

    def change_styles(self, styles, n_frames, change_mode, inversed=False):
        assert isinstance(change_mode, StyleChangeMode), change_mode
        styles = [s.unsqueeze(1).repeat(1, n_frames, 1) for s in styles]  # [bs x n_frames x code_dim] * mixin_num

        if inversed:
            dyn_style_coordinates = self.code_dim - self.dyn_style_coordinates
        else:

            dyn_style_coordinates = self.dyn_style_coordinates

        if (dyn_style_coordinates == 0) or (change_mode is StyleChangeMode.REPEAT):
            return styles

        batch_size = styles[0].shape[0]
        device = styles[0].device

        if change_mode is StyleChangeMode.INTERPOLATE:
            dyn1 = [torch.randn(batch_size, 1, dyn_style_coordinates, device=device)] * len(styles)
            dyn2 = [torch.randn(batch_size, 1, dyn_style_coordinates, device=device)] * len(styles)
            delta_mult = torch.linspace(0., 1., n_frames, device=device)
            delta_mult = delta_mult[None, :, None]
            delta_mult = delta_mult * constants.STYLE_CHANGE_COEF
            dyn = []
            for i in range(len(styles)):
                dyn.append(slerp(delta_mult, dyn1[i], dyn2[i]))
        elif change_mode is StyleChangeMode.RESAMPLE:
            dyn = [torch.randn(batch_size, n_frames, dyn_style_coordinates, device=device)] * len(styles)

        new_styles = []
        for i, s in enumerate(styles):
            if inversed:
                concatenated = torch.cat([dyn[i], s[:, :, dyn_style_coordinates:]], dim=2)
            else:
                concatenated = torch.cat([s[:, :, :-dyn_style_coordinates], dyn[i]], dim=2)
            new_styles.append(concatenated)
        return new_styles

    def get_styles(self, latents, mean_style, style_weight, n_frames, change_mode, inversed=False):
        if type(latents) not in (list, tuple):
            latents = [latents]

        # change z
        latents = self.change_styles(latents, n_frames, change_mode, inversed)
        batch_size = latents[0].shape[0]
        latents = frames2batch(latents)

        # mlp
        styles = []
        for z in latents:
            styles.append(self.style(z))

        # truncation trick
        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))
            styles = styles_norm

        styles = batch2frames(styles, batch_size, n_frames)
        return styles

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
        latent_type='z',
        n_frames=1,
        style_change_mode=StyleChangeMode.RESAMPLE,
        noise_change_modes=tuple([NoiseChangeMode.RESAMPLE]*constants.MAX_LAYERS_NUM),
        inversed=False,
        homographies=None,
    ):
        if latent_type == 'z':
            styles = self.get_styles(input, mean_style, style_weight, n_frames,
                                     style_change_mode, inversed)
        elif latent_type == 'w':
            styles = input
        else:
            raise ValueError

        batch_size = styles[0].shape[0]

        if noise is None:
            device = input[0].device
            noise = self.get_noise(batch_size, n_frames, step, device, noise_change_modes,
                                   inversed, homographies)

        styles = frames2batch(styles)
        noise = frames2batch(noise)
        fake_image = self.generator(styles, noise, step, alpha, mixing_range=mixing_range)
        fake_image = batch2frames(fake_image, batch_size, n_frames)

        return fake_image

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)
        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False, n_frames=1):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channels):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channels, 1), nn.LeakyReLU(0.2))
            else:
                return EqualConv2d(3, out_channels, 1)

        from_rgb_out_channels_num = [16, 32, 64, 128, 256, 512, 512, 512, 512]
        self.from_rgb = nn.ModuleList(
            [make_from_rgb(out_channels) for out_channels in from_rgb_out_channels_num]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        features = self.get_features(input, step, alpha)
        out = self.linear(features)
        return out

    def get_features(self, input, step=0, alpha=-1):
        batch_size, n_frames = input.shape[:2]
        input = frames2batch(input)
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                frames_out = batch2frames(out, batch_size, n_frames)
                out_std = torch.sqrt(frames_out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean().expand_as(out[:, [0]])
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(-1).squeeze(-1)
        return out


class NFramesDiscriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False, n_frames=2,
                 channels=constants.NFD_CHANNELS):
        super().__init__()

        progression = []
        from_rgb_out_channels_num = []
        for i in range(9):
            from_rgb_out_channels_num.append(channels[i] // n_frames)
            if i <= 3:  # 1024 - 128
                progression.append(ConvBlock(channels[i], channels[i + 1], 3, 1, 3, 1, downsample=True, fused=fused))
            elif i < 8:  # 64
                progression.append(ConvBlock(channels[i], channels[i + 1], 3, 1, 3, 1, downsample=True))
            else:  # 4
                progression.append(ConvBlock(channels[i] + 1, channels[i + 1], 3, 1, 4, 0))

        self.progression = nn.ModuleList(progression)

        def make_from_rgb(out_channels):
            in_channels = 3
            module = EqualConv2d(in_channels, out_channels, 1)
            if from_rgb_activate:
                return nn.Sequential(module, nn.LeakyReLU(0.2))
            else:
                return module

        self.from_rgb = nn.ModuleList(
            [make_from_rgb(out_channels) for out_channels in from_rgb_out_channels_num]
        )

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(channels[-1], 1)

    def forward(self, input, step=0, alpha=-1):
        out = self.linear(self.get_features(input, step, alpha))
        return out

    def get_features(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                frames = [self.from_rgb[index](input[:, f_i]) for f_i in range(input.shape[1])]
                out = torch.cat(frames, dim=1)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean().expand(out.shape[0], 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    bs, n_frames = input.shape[:2]
                    pooled = F.avg_pool2d(input.view(bs * n_frames, *input.shape[2:]), 2)
                    pooled = pooled.view(bs, n_frames, *pooled.shape[1:])
                    frames = [self.from_rgb[index + 1](pooled[:, f_i]) for f_i in range(pooled.shape[1])]
                    skip_rgb = torch.cat(frames, dim=1)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(-1).squeeze(-1)
        return out
