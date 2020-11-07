import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ade20k_segm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss


class LMaskLoss(nn.Module):
    def __init__(self, l_pix_type, segm_mask):
        super().__init__()
        self.segm_mask = segm_mask
        if l_pix_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif l_pix_type == 'l2':
            self.loss_func = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
        self.segmentor = psina_seg.base.SegmentationModule(encode='stationary_probs')#'stationary_probs')

    def forward(self, pred, hr, ref):
        self.segmentor = self.segmentor.eval()
        lr = F.interpolate(hr, scale_factor=0.25, mode='nearest')
        with torch.no_grad():
            if self.segm_mask[0] == -1:
                assert len(self.segm_mask) == 1
                binary_mask = (1 - self.segmentor.predict(hr, imgSizes=self.segm_mask))
            else:
                image = F.interpolate(hr, scale_factor=0.5, mode='nearest')
                binary_mask = (1 - self.segmentor.predict(image, imgSizes=self.segm_mask))
                binary_mask = F.interpolate(binary_mask, scale_factor=2, mode='nearest')

        res = 0
        for i in range(binary_mask.shape[1]):
            stationary = self.loss_func(binary_mask[:,i,::].unsqueeze(1) * ref, binary_mask[:,i,::].unsqueeze(1) * pred)
            dynamic = self.loss_func((1-binary_mask[:,i,::].unsqueeze(1)) * hr, (1-binary_mask[:,i,::].unsqueeze(1)) * pred)
            res += stationary + dynamic
        return res
