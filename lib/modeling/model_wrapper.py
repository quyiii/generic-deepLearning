import torch
import numpy as np
from torch import nn
from .models import get_model_class

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self)
        self.cfg = cfg
        prepare_model()

    def prepare_model(self):
        model_class = get_model_class(self.cfg)
        if self.cfg.

    def forward(self, x):
        return self.model(x)

    def get_loss(self):
        loss = None
        return loss
    