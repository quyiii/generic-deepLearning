from . import digit_models
from . import image_models
from . import video_models
from . import word_models
import importlib
import functools
import torch.nn as nn 

def get_model_class(cfg):
    file_name = 'lib.modeling.models.'+cfg.INPUT.TYPE.lower()+'_models'
    modelslib = importlib.import_module(file_name)

    model = None
    target_model_name = cfg.MODEL.NAME.replace('_', '')
    for name, cls in modelslib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module)
            model = cls

    if model is None:
        raise NotImplementError("In {}.py, there is no class {}".format(cfg.INPUT.TYPE.lower()+'_models', target_model_name))
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
        raise NotImplementError('normalization layer {} is not implemented'.format(norm_layer))
    return norm_layer