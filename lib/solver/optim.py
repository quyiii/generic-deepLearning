import torch

def get_optim(cfg, params):
    optim_name = cfg.SOLVER.OPTIM_NAME
    optim = None
    if optim_name == 'Adam':
        optim = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR,
                                 betas=(cfg.SOLVER.OPTIM_BETA if cfg.SOLVER.OPTIM_BETA.lower() != 'none' else 0.9, 0.999))
    else:
        raise NotImplementedError("optim {} is not implemented".format(optim_name))
    return optim