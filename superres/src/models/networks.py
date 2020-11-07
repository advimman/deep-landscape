import logging

import torch

import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.video_discriminator as video_discriminator

logger = logging.getLogger('base')


def define_G(opt, name='network_G'):
    opt_net = opt[name]
    which_model = opt_net['which_model_G']

    if which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'RRDBNetSEG':
        netG = RRDBNet_arch.RRDBNetSEG(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], segm_mask=opt['train']['segm_mask'],
                                       scale=opt_net['scale'])     
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_128_mask':
        netD = SRGAN_arch.Discriminator_VGG_128_MASK(in_nc=opt_net['in_nc'], nf=opt_net['nf'], w=opt_net['classic_disc_w_path'])
    elif which_model == 'discriminator_vgg_24':
        netD = SRGAN_arch.Discriminator_VGG_24(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def define_video_D(opt):
    opt_net = opt.get("network_video_D")
    if opt_net is None:
        return None
    which_model = opt_net['which_model_D']

    if which_model == 'resnet34':
        netD = video_discriminator.resnet34()
    elif which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128_Video(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def define_F(opt, use_bn=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
