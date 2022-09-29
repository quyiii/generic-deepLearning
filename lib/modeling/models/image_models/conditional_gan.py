import torch
import torch.nn as nn
from .conditional_gan_nets import get_G, get_D
from lib.solver import get_loss_class, get_optim

class ConditionalGan(nn.Module):
    def __init__(self, cfg):
        super(ConditionalGan, self).__init__()
        self.cfg = cfg
        self.direction = cfg.INPUT.DIRECTION
        self.device = torch.device("cuda:{}".format(cfg.MODEL.DEVICE_IDS[0])) if cfg.MODEL.DEVICE.lower() == 'cuda' else torch.device("cpu")
        self.istrain = cfg.TRAIN.IS_TRAIN
        input_channel = cfg.INPUT.CHANNEL if self.direction else cfg.OUTPUT.CHANNEL
        output_channel = cfg.OUTPUT.CHANNEL if self.direction else cfg.INPUT.CHANNEL
        G_type = cfg.MODEL.CONSIST.G
        D_type = cfg.MODEL.CONSIST.D
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        self.model_names = ['G', 'D'] if self.istrain else ['G']
        
        self.netG = get_G(input_channel, output_channel, 64, G_type, cfg.MODEL.NORM, cfg.MODEL.DROPOUT, cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN)

        if self.istrain:
            # the input of D is cat(x, y)
            self.netD = get_D(input_channel + output_channel, 64, D_type, cfg.MODEL.NORM, cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN)
            # 自己设计的要放在cuda上
            self.criterionGAN = get_loss_class(cfg, 0)('lsgan').to(self.device)
            # 自带的无需cuda
            self.criterionL1 = get_loss_class(cfg, 1)()
            self.optimizer_G = get_optim(cfg, self.netG.parameters())
            self.optimizer_D = get_optim(cfg, self.netD.parameters())
        
    def set_input(self, input):
        # if batch_size > 1, A means As
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_path = input['AB_path']
    
    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * .5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.cfg.LOSS.LAMBDA_L1
        self.loss_G.backward()
        return self.loss_G.item()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        loss = self.backward_G()
        self.optimizer_G.step()
        return loss