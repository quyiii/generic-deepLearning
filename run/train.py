import sys
import argparse

sys.path.append('.')
from lib.config import cfg

def get_args():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    return parser.parse_args()

def get_cfg(config_file):
    if config_file != "":
        cfg.merge_from_file(config_file)
    cfg.freeze()

def main():
    args = get_args()
    get_cfg(args.config_file)
    print(cfg)

if __name__ == '__main__':
    main()