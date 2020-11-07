import numpy as np
import torch
import torch.nn.functional as F


class SSIM(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, window_size=11, size_average=True, is_video=False, order='bcthw'):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
        self.is_video = is_video
        self.order = order
        if self.is_video:
            if order == 'bcthw':
                self.default_order = True
            elif order == 'btchw':
                self.default_order = False
            else:
                raise NotImplementedError("use 'bcthw' or 'btchw' channels order for video")

    def forward(self, img1, img2):
        assert len(img1.shape) == 5 if self.is_video else 4

        if not self.is_video:
            channel = img1.size()[1]
        elif self.order == 'bcthw':
            channel = img1.size()[1]
        elif self.order == 'btchw':
            channel = img1.size()[2]
        else:
            raise NotImplementedError()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            window = window.to(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        if self.is_video:
            if self.order == 'bcthw':
                img1.transpose_(1, 2)
                img2.transpose_(1, 2)

            img1 = img1.reshape(img1.shape[0] * img1.shape[1], img1.shape[2], img1.shape[3], img1.shape[4])
            img2 = img2.reshape(img2.shape[0] * img2.shape[1], img2.shape[2], img2.shape[3], img2.shape[4])

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - (window_size // 2)) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=(window_size // 2), groups=channel)
        mu2 = F.conv2d(img2, window, padding=(window_size // 2), groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=(window_size // 2), groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=(window_size // 2), groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=(window_size // 2), groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()

        return ssim_map.mean(1).mean(1).mean(1)
