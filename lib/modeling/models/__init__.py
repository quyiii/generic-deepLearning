from . import digit_models
from . import image_models
from . import video_models
from . import word_models
import importlib
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

    
