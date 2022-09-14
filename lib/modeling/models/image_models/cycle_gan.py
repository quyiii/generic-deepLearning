import torch.nn as nn

class CycleGan(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, x):