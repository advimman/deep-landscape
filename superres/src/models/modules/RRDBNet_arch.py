import os
import sys
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.transforms import Normalize
from guided_filter_pytorch.guided_filter import GuidedFilter

import models.modules.module_util as mutil
from models.modules import unet
from models.modules import ade20k_segm
from models.modules import style


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x



class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward_without_last_conv(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.lrelu(self.HRconv(fea))
        return out

    def forward(self, x):
        out = self.forward_without_last_conv(x)
        out = self.conv_last(out)

        return out, None


class RRDBNetSEG(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, segm_mask, scale=4, gc=32):
        super().__init__()
        self.nf = nf
        self.binary_mask = ade20k_segm.base.SegmentationModule(encode='stationary_not_dynamic_probs').eval() #stationary_not_dynamic_probs stationary_probs
        self.scale = scale
        if scale == 4:
            self.net = RRDBNet(in_nc, out_nc, nf, nb, gc)
        elif scale == 8:
            self.net = RRDBNetX8(in_nc, out_nc, nf, nb, gc)
        else:
            raise NotImplementedError
        self.net = self.net.eval()

        self.conv_ref = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=False)
        self.enhancement1 = nn.Conv2d(nf*2*2, nf, 3, 1, 1, bias=False)
        self.enhancement2 = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

        self.binary_mask_upconv = nn.Conv2d(1, 1, 3, 1, 1, bias=True)
        self.drop = nn.Dropout2d(p=0.2)

        self.segm_mask = segm_mask

        guided_filter_r = 10 #7
        guided_eps = 1e-3 #3e-4
        self.train_gdf = GuidedFilter(r=guided_filter_r, eps=guided_eps)
        self.test_gdf = GuidedFilter(r=guided_filter_r*2, eps=guided_eps)
        self.pool_gdf = nn.AdaptiveAvgPool2d((4,4))
        self.linear_gdf1 = nn.Linear(16 * self.nf, 16 * self.nf)
        self.linear_gdf2 = nn.Linear(16 * self.nf, 2 * self.nf)

    def load_pretrained_net_weights(self, model_path):
        checkpoint = torch.load(model_path)
        if self.scale == 4:
            self.net.load_state_dict(checkpoint)
        elif self.scale == 8:
            self.net.load_state_dict(checkpoint)

    def _get_gdf_params(self, ref):
        x = self.pool_gdf(ref).view(-1, 16*self.nf)
        x = F.relu(self.linear_gdf1(x))
        x = nn.Sigmoid()(self.linear_gdf2(x))
        return x

    def get_binary_mask(self, image):
        self.binary_mask = self.binary_mask.eval()
        with torch.no_grad():
            if self.segm_mask[0] == -1:
                assert len(self.segm_mask) == 1
                binary_mask = (1-self.binary_mask.predict(image, imgSizes=self.segm_mask))
            else:
                # self.binary_mask 1 is dynamic and 0 is static
                image = F.interpolate(image, scale_factor=0.5, mode='nearest')
                binary_mask = (1-self.binary_mask.predict(image, imgSizes=self.segm_mask))
                binary_mask = F.interpolate(binary_mask, scale_factor=2, mode='nearest')
        return binary_mask

    def forward(self, x, ref, aux=None):
        with torch.no_grad():
            feat = self.net.forward_without_last_conv(x)
        binary_mask = self.get_binary_mask(ref)
        ref = self.lrelu(self.conv_ref(ref))

        # Guided simple
        if self.training:
            ref = self.train_gdf(ref, feat)
        else:
            ref = self.test_gdf(ref, feat)
        
        concated = []
        for i in range(2):
            masked_ref = (binary_mask[:,i,::].unsqueeze(1))*ref
            masked_feat = (1-binary_mask[:,i,::].unsqueeze(1))*feat
            concated += [masked_ref, masked_feat]

        concated = torch.cat(concated, dim=1)
        res = self.enhancement1(self.lrelu(concated))
        res = self.enhancement2(self.lrelu(res))
        return res, binary_mask