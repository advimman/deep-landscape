import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, LMaskLoss

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.train_opt = train_opt
        self.opt = opt

        self.segmentor = None

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if train_opt.get("gan_video_weight", 0) > 0:
                self.net_video_D = networks.define_video_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
                if train_opt.get("gan_video_weight", 0) > 0:
                    self.net_video_D = DistributedDataParallel(self.net_video_D,
                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)
                if train_opt.get("gan_video_weight", 0) > 0:
                    self.net_video_D = DataParallel(self.net_video_D)

            self.netG.train()
            self.netD.train()
            if train_opt.get("gan_video_weight", 0) > 0:
                self.net_video_D.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # Pixel mask loss
            if train_opt.get("pixel_mask_weight", 0) > 0:
                l_pix_type = train_opt['pixel_mask_criterion']
                self.cri_pix_mask = LMaskLoss(l_pix_type=l_pix_type, segm_mask=train_opt['segm_mask']).to(self.device)
                self.l_pix_mask_w = train_opt['pixel_mask_weight']
            else:
                logger.info('Remove pixel mask loss.')
                self.cri_pix_mask = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # Video gan weight
            if train_opt.get("gan_video_weight", 0) > 0:
                self.cri_video_gan = GANLoss(train_opt['gan_video_type'], 1.0, 0.0).to(self.device)
                self.l_gan_video_w = train_opt['gan_video_weight']  

                # can't use optical flow with i and i+1 because we need i+2 lr to calculate i+1 oflow
                if 'train' in self.opt['datasets'].keys():
                    key = "train"
                else:
                    key = 'test_1'
                assert self.opt['datasets'][key]['optical_flow_with_ref'] == True, f"Current value = {self.opt['datasets'][key]['optical_flow_with_ref']}"
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # Video D
            if train_opt.get("gan_video_weight", 0) > 0:
                self.optimizer_video_D = torch.optim.Adam(self.net_video_D.parameters(), lr=train_opt['lr_D'],
                                                    weight_decay=wd_D,
                                                    betas=(train_opt['beta1_D'], train_opt['beta2_D']))
                self.optimizers.append(self.optimizer_video_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.img_path = data['GT_path']
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
        if self.train_opt.get("use_HR_ref"):
            self.var_HR_ref = data['img_reference'].to(self.device) 
        if "LQ_next" in data.keys():
            self.var_L_next = data['LQ_next'].to(self.device)
            if "GT_next" in data.keys():
                self.var_H_next = data['GT_next'].to(self.device)
                self.var_video_H = torch.cat([data['GT'].unsqueeze(2), data['GT_next'].unsqueeze(2)], dim=2).to(self.device)
        else:
            self.var_L_next = None

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        args = [self.var_L]
        if self.train_opt.get('use_HR_ref'):
            args += [self.var_HR_ref]
        if self.var_L_next is not None:
            args += [self.var_L_next]
        self.fake_H, self.binary_mask = self.netG(*args)

        #Video Gan
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            with torch.no_grad():
                args = [self.var_L, self.var_HR_ref, self.var_L_next]
                self.fake_H_next, self.binary_mask_next = self.netG(*args)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_pix_mask:
                l_g_pix_mask = self.l_pix_mask_w * self.cri_pix_mask(self.fake_H, self.var_H, self.var_HR_ref)
                l_g_total += l_g_pix_mask               
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea

            # Image Gan
            if self.opt['network_D'] == "discriminator_vgg_128_mask":
                import torch.nn.functional as F
                from models.modules import psina_seg
                if self.segmentor is None:
                    self.segmentor = psina_seg.base.SegmentationModule(encode='stationary_probs').to(self.device)
                self.segmentor = self.segmentor.eval()
                lr = F.interpolate(self.var_H, scale_factor=0.25, mode='nearest')
                with torch.no_grad():
                    binary_mask = (1 - self.segmentor.predict(lr[:, [2,1,0],::]))
                binary_mask = F.interpolate(binary_mask, scale_factor=4, mode='nearest')
                pred_g_fake = self.netD(self.fake_H, self.fake_H *(1-binary_mask), self.var_HR_ref, binary_mask * self.var_HR_ref)
            else:
                pred_g_fake = self.netD(self.fake_H)

            if self.opt['train']['gan_type'] == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                if self.opt['network_D'] == "discriminator_vgg_128_mask":
                    pred_g_fake = self.netD(self.var_H, self.var_H *(1-binary_mask), self.var_HR_ref, binary_mask * self.var_HR_ref)                    
                else:
                    pred_d_real = self.netD(self.var_H)
                pred_d_real = pred_d_real.detach()
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan


            #Video Gan
            if self.opt['train'].get("gan_video_weight", 0) > 0:
                self.fake_video_H = torch.cat([self.fake_H.unsqueeze(2), self.fake_H_next.unsqueeze(2)], dim=2)
                pred_g_video_fake = self.net_video_D(self.fake_video_H)
                if self.opt['train']['gan_video_type'] == 'gan':
                    l_g_video_gan = self.l_gan_video_w * self.cri_video_gan(pred_g_video_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    pred_d_video_real = self.net_video_D(self.var_video_H)
                    pred_d_video_real = pred_d_video_real.detach()
                    l_g_video_gan = self.l_gan_video_w * (
                        self.cri_video_gan(pred_d_video_real - torch.mean(pred_g_video_fake), False) +
                        self.cri_video_gan(pred_g_video_fake - torch.mean(pred_d_video_real), True)) / 2
                l_g_total += l_g_video_gan

            # OFLOW regular
            if self.binary_mask is not None:
                l_g_total += 1* self.binary_mask.mean()

            l_g_total.backward()
            self.optimizer_G.step()

         # D
        for p in self.netD.parameters():
            p.requires_grad = True
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            for p in self.net_video_D.parameters():
                p.requires_grad = True

        # optimize Image D
        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_H)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        if self.opt['train']['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2
        l_d_total.backward()
        self.optimizer_D.step()

        # optimize Video D
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            self.optimizer_video_D.zero_grad()
            l_d_video_total = 0
            pred_d_video_real = self.net_video_D(self.var_video_H)
            pred_d_video_fake = self.net_video_D(self.fake_video_H.detach())  # detach to avoid BP to G
            if self.opt['train']['gan_video_type'] == 'gan':
                l_d_video_real = self.cri_video_gan(pred_d_video_real, True)
                l_d_video_fake = self.cri_video_gan(pred_d_video_fake, False)
                l_d_video_total = l_d_video_real + l_d_video_fake
            elif self.opt['train']['gan_video_type'] == 'ragan':
                l_d_video_real = self.cri_video_gan(pred_d_video_real - torch.mean(pred_d_video_fake), True)
                l_d_video_fake = self.cri_video_gan(pred_d_video_fake - torch.mean(pred_d_video_real), False)
                l_d_video_total = (l_d_video_real + l_d_video_fake) / 2
            l_d_video_total.backward()
            self.optimizer_video_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            self.log_dict['D_video_real'] = torch.mean(pred_d_video_real.detach())
            self.log_dict['D_video_fake'] = torch.mean(pred_d_video_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            args = [self.var_L]
            if self.train_opt.get('use_HR_ref'):
                args += [self.var_HR_ref]
            if self.var_L_next is not None:
                args += [self.var_L_next]
            self.fake_H, self.binary_mask = self.netG(*args)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if self.binary_mask is not None:
            out_dict['binary_mask'] = self.binary_mask.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        # G
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['pretrain_model_G_strict_load'])

        if self.opt['network_G'].get("pretrained_net") is not None:
            self.netG.module.load_pretrained_net_weights(self.opt['network_G']['pretrained_net'])

        # D
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['pretrain_model_D_strict_load'])

        # Video D
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            load_path_video_D = self.opt['path'].get("pretrain_model_video_D")
            if self.opt['is_train'] and load_path_video_D is not None:
                self.load_network(load_path_video_D, self.net_video_D, self.opt['path']['pretrain_model_video_D_strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)
        if self.opt['train'].get("gan_video_weight", 0) > 0:
            self.save_network(self.net_video_D, 'video_D', iter_step)

    @staticmethod
    def _freeze_net(network):
        for p in network.parameters():
            p.requires_grad = False
        return network

    @staticmethod
    def _unfreeze_net(network):
        for p in network.parameters():
            p.requires_grad = True
        return network

    def freeze(self, G, D):
        if G:
            self.netG.module.net = self._freeze_net(self.netG.module.net)
        if D:
            self.netD.module = self._freeze_net(self.netD.module)

    def unfreeze(self, G, D):
        if G:
            self.netG.module.net = self._unfreeze_net(self.netG.module.net)
        if D:
            self.netD.module = self._unfreeze_net(self.netD.module)
