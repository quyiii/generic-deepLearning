import torch
import torch.nn as nn
from .conditional_gan_nets import get_G, get_D
from lib.solver import get_loss_class, get_optim

class ConditionalGan(nn.Module):
    def __init__(self, cfg):
        super(ConditionalGan, self).__init__()
        self.cfg = cfg
        self.direction = cfg.INPUT.DIRECTION
        self.device = torch.device("cuda:{}".format(cfg.MODEL.DEVICE_IDS[0])) if self.MODEL.DEVICE.lower() == 'cuda' else torch.device("cpu")
        self.istrain = self.TRAIN.IS_TRAIN
        input_channel = cfg.INPUT.CHANNEL if self.direction else cfg.OUTPUT.CHANNEL
        output_channel = cfg.OUTPUT.CHANNEL if self.direction else cfg.INPUT.CHANNEL
        G_type = cfg.MODEL.CONSIST.G
        D_type = cfg.MODEL.CONSIST.D
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        self.model_names = ['G', 'D'] if self.istrain else ['G']
        
        self.netG = get_G(input_channel, output_channel, 64, G_type, cfg.MODEL.NORM, cfg.MODEL.DROPOUT, cfg.MDOEL.INIT, cfg.MDOEL.INIT_GAIN)

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
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_path = input[]