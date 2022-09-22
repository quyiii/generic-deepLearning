from .model_wrapper import ModelWrapper

import torch
import torch.nn as nn

def create_model(cfg):
    return parallel(ModelWrapper(cfg), cfg)

def parallel(model, cfg):
    device = cfg.MODEL.DEVICE.lower()
    gpu_ids = cfg.MODEL.DEVICE_IDS
    if device == 'cuda':
        assert(torch.cuda.is_available())
        gpu_num = len(gpu_ids)
        if gpu_num == 0:
            raise RuntimeError('no gpu_ids for run program')
        elif gpu_num >=2:
            model = nn.DataParallel(model, gpu_ids)
        # to() 输入数字a 默认为cuda:a
        model.to("cuda:{}".format(gpu_ids[0]))
    elif device == 'cpu':
        model.to('cpu')
    else: 
        raise NotImplementedError('no device {}'.format(device))
    return model