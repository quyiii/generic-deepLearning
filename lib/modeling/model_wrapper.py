import torch
import numpy as np
from torch import nn
from .models import get_model_class

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.cfg = cfg
        self.model_parallel = None
        self.model_name = cfg.MODEL.NAME.lower()
        self._prepare_model()

    def _prepare_model(self):
        model_class = get_model_class(self.cfg)
        if self.model_name in ['cycle_gan', 'conditional_gan']:
            self.model = model_class(self.cfg)
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))
        self.parallel()

    def parallel(self):
        device = self.cfg.MODEL.DEVICE.lower()
        gpu_ids = self.cfg.MODEL.DEVICE_IDS
        if device == 'cuda':
            assert(torch.cuda.is_available())
            gpu_num = len(gpu_ids)
            if gpu_num == 0:
                raise RuntimeError('no gpu_ids for run program')
            elif gpu_num >=2:
                self.model_parallel = nn.DataParallel(self, gpu_ids)
            # to() 输入数字a 默认为cuda:a
            self = self.to("cuda:{}".format(gpu_ids[0]))
        elif device == 'cpu':
            self = self.to('cpu')
        else: 
            raise NotImplementedError('no device {}'.format(device))

    def forward(self, x):
        if self.model_name in ['cycle_gan', 'conditional_gan']:
            self.model.set_input(x)
            self.model()
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))

    def optimize_parameters(self, x):
        loss = None
        if self.model_name in ['cycle_gan', 'conditional_gan']:
            self.model.set_input(x)
            loss = self.model.optimize_parameters()
        else:
            raise NotImplementedError("model {} is not implemented".format(self.model_name))
        return loss

    def test(self, x):
        self.eval()
        with torch.no_grad:
            if self.model_name == 'cycle_gan':
                self.forward(x)

