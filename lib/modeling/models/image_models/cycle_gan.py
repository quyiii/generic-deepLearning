import torch
import itertools
import torch.nn as nn
from .cycle_gan_nets import get_G, get_D
from lib.solver import get_loss_class, get_optim

'''
This model named cyle_gan, which is used to transform the image's style

Input: origin image 
Output: style-transformed image  

Generator G_A: fake_B = G_A(A)
Generator G_B: fake_A = G_B(B)

Discriminator D_A: D_A(B) vs D_A(fake_B)
Discriminator D_B: D_B(A) vs D_B(fake_A)

loss D_A: loss of Discriminator D_A
loss G_A: loss of Generator G_A ==> D_A 
loss cycle_A: lambda_A * ||G_B(G_A(A)) - A||
loss idt_A: lambda_identity * ||G_A(B) - B|| * lambda_B

loss D_B: loss of Discriminator D_B
loss G_B: loss of Generator G_B ==> D_B
loss cycle_B: lambda_B * ||G_A(G_B(B)) - B||
loss idt_B: lambda_identity * ||G_B(A) - A|| * lambda_A

'''

class CycleGan(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:{}".format(cfg.MODEL.DEVICE_IDS[0])) if self.MODEL.DEVICE.lower() == 'cuda' else torch.device('cpu')
        self.direction = cfg.INPUT.DIRECTION
        self.is_train = self.cfg.TRAIN.IS_TRAIN

        self.netG_A = get_G(cfg.INPUT.CHANNEL, cfg.OUTPUT.CHANNEL, 64, cfg.MODEL.CONSIST.G, cfg.MODEL.NORM,
                            cfg.MODEL.DROPOUT, cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN, cfg.DEVICE_IDS)
        self.netG_B = get_G(cfg.OUTPUT.CHANNEL, cfg.INPUT.CHANNEL, 64, cfg.MODEL.CONSIST.G, cfg.MODEL.NORM,
                            cfg.MODEL.DROPOUT, cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN, cfg.DEVICE_IDS)

        if not self.is_train:
            self.model_names = ['G_A', 'G_B']
        else:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B'] 
            self.netD_A = get_D(cfg.OUTPUT.CHANNEL, 64, cfg.MODEL.CONSIST.D, cfg.MODEL.NORM,
                                cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN, cfg.MODEL.DEVICE_IDS)
            self.netD_B = get_D(cfg.INPUT.CHANNEL, 64, cfg.MODEL.CONSIST.D, cfg.MODEL.NORM,
                                cfg.MODEL.INIT, cfg.MODEL.INIT_GAIN, cfg.MODEL.DEVICE_IDS)
            if cfg.LOSS.LAMBDA_IDENTITY > 0:
                assert(cfg.INPUT.CHANNEL == cfg.OUTPUT.CHANNEL)
            
            self.criterionGAN = get_loss_class(cfg, 0)('lsgan').to(self.device)
            self.criterionCycle = get_loss_class(cfg, 1)()
            self.criterionIdt = get_loss_class(cfg, 1)() 

            self.optimizer_G = get_optim(cfg, itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
            self.optimizer_D = get_optim(cfg, itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()))

    def set_input(self, data):
        self.real_A = data['A' if self.direction else 'B'].to(self.device)
        self.real_B = data['B' if self.direction else 'A'].to(self.device)
        self.iamge_paths = data['A_paths' if self.direction else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_baisc(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # fake
        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # combine loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = 