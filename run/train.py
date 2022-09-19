import os
import sys
import torch
import argparse

sys.path.append('.')
from lib.config import cfg
from lib.solver import get_loss_class

def get_args():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--train', action='store_true', help='choose train or test')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    return parser.parse_args()

def get_cfg(args):
    config_file = args.config_file
    if config_file != "":
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(['TRAIN.IS_TRAIN', args.train])
        cfg.SOLVER.START_EPOCH = get_start_epoch(cfg)
    # cfg.defrost() 解冻
    cfg.freeze() 

def get_start_epoch(cfg):
    start_epoch = 0
    if cfg.TRAIN.IS_TRAIN and cfg.CHECKPOINT.RESUME != 'none':
        if not os.path.isfile(cfg.CHECKPOINT.RESUME):
            raise RuntimeError("not find checkpoint {}".format(cfg.CHECKPOINT.RESUME))
        checkpoint = torch.load(cfg.CHECKPOINT.RESUME)
        start_epoch = checkpoint['epoch']
    return start_epoch

def main():
    args = get_args()
    get_cfg(args)
    print(cfg)

if __name__ == '__main__':
    main()
    print(cfg.SOLVER.START_EPOCH)
    loss = get_loss_class(cfg, 1)
    print(loss.__name__)