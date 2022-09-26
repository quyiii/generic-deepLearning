import torch
import torch.nn as nn
from .conditional_gan_nets import get_G, get_D


class ConditionalGan(nn.Module):
    def __init__(self, cfg):
        super(ConditionalGan, self).__init__()
        self.cfg = cfg
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        if cfg.TRAIN.IS_TRAIN:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = 
