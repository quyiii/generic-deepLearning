import torch
import numpy as np
from torch import nn
from .models import get_model_class

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.cfg = cfg
        self.model_name = cfg.MODEL.NAME.lower()
        self._prepare_model()

    def _prepare_model(self):
        model_class = get_model_class(self.cfg)
        if self.model_name == 'cycle_gan':
            self.base_model = model_class(self.cfg)
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))

    def forward(self, x):
        if self.model_name == 'cycle_gan':
            self.base_model.set_input(x)
            self.base_model()
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))

    def optimize_parameters(self, x):
        loss = None
        if self.model_name == 'cycle_gan':
            self.base_model.set_input(x)
            loss = self.base_model.optimize_parameters()
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))
        return loss

    def test(self, x):
        self.eval()
        with torch.no_grad:
            if self.model_name == 'cycle_gan':
                self.forward(x)

