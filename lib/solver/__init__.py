from .lr_scheduler import get_scheduler
from .optim import get_optim
import importlib
import torch.nn as nn 

def get_loss_class(cfg, index=0):
    file_name = 'lib.solver.losses.'+cfg.OUTPUT.TYPE.lower()+'_losses'
    losseslib = importlib.import_module(file_name)

    loss = None
    loss_names = cfg.LOSS.NAME
    if not isinstance(loss_names, list):
        loss_names = [loss_names]
    target_loss_name = loss_names[index].lower().replace('_', '')
    for name, cls in losseslib.__dict__.items():
        if name.lower() == target_loss_name.lower() \
           and issubclass(cls, nn.Module):
            loss = cls
            break

    if loss is None:
        raise NotImplementedError("In {}, class {} is not implemented".format(cfg.OUTPUT.TYPE.lower()+'_losses', target_loss_name))
    return loss