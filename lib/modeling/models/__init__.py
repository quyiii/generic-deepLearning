import importlib
import functools
import torch
import torch.nn as nn 
from torch.nn import init

def get_model_class(cfg):
    file_name = 'lib.modeling.models.'+cfg.INPUT.TYPE.lower()+'_models'
    modelslib = importlib.import_module(file_name)

    model = None
    target_model_name = cfg.MODEL.NAME.replace('_', '')
    for name, cls in modelslib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        raise NotImplementedError("In {}.py, class {} is not implemented".format(cfg.INPUT.TYPE.lower()+'_models', target_model_name))
    return model

class NoneNorm(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='none'):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        # functools.partial 提前指定某函数的一些参数，使得该函数的参数量变少
        # batchNorm2d 给 b*c*h*w 的图像数据的每一个channel进行一次标准化
        # affine 是否学习gamma beta ，是标准化的weight 和 bias
        # track_running_stats 需不需要更新均值 方差
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return NoneNorm()
    else:
        raise NotImplementedError('normalization layer {} is not implemented'.format(norm_layer))
    return norm_layer

def init_net(net, init_type='normal', init_gain=0.02, device='cuda', gpu_ids=[0]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if device.lower() == 'cuda':
        assert(torch.cuda.is_available())
        if len(gpu_ids) == 0:
            raise RuntimeError('no gpu_ids for run program')
        net = nn.DataParallel(net, gpu_ids)
        net.to("cuda:" + gpu_ids[0])
    elif device.lower() == 'cpu':
        net.to('cpu')
    else: 
        raise NotImplementedError('no device {}'.format(device))
    init_weight(net, init_type, init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization mothod {} is not implemented'.format(norm_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    # 模型的所有模块都要初始化
    net.apply(init_func)
