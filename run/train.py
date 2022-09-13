import sys
import argparse

sys.path.append('.')
from lib.config import cfg

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
    cfg.freeze()

def main():
    args = get_args()
    get_cfg(args)
    print(cfg)

if __name__ == '__main__':
    main()
    print(cfg.PROCESS.FLIP_P)