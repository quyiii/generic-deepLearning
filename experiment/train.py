import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append('.')
from lib.config import cfg
from lib.util import creat_saver_writer
from lib.solver import get_loss_class
from lib.modeling import create_model
from lib.data import get_dataLoader
from lib.data.image_datasets import tensorToPIL

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
        cfg.TRAIN.START_EPOCH = get_start_epoch(cfg)
    # cfg.defrost() 解冻
    cfg.freeze() 

def get_start_epoch(cfg):
    start_epoch = 0
    if cfg.TRAIN.IS_TRAIN and cfg.CHECKPOINT.RESUME != 'none':
        if not os.path.isfile(cfg.CHECKPOINT.RESUME):
            raise RuntimeError("not find checkpoint file {}".format(cfg.CHECKPOINT.RESUME))
        checkpoint = torch.load(cfg.CHECKPOINT.RESUME)
        start_epoch = checkpoint['epoch'] + 1
    return start_epoch

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.saver, self.writer = creat_saver_writer(cfg)

        self.model = create_model(cfg)

        self.dataloader = get_dataLoader(cfg)
    
    def train(self, epoch):
        # tbar = tqdm(self.dataloader)
        loss = []
        for i, data in enumerate(self.dataloader):
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            loss.append(self.model.optimize_parameters(data))
        imgs = self.model.base_model.get_current_visuals()
        # img_a = tensorToPIL(imgs['real_A'])
        # img_ab = tensorToPIL(imgs['fake_B'])
        self.writer.add_scalar('loss', np.array(loss).mean(), epoch)
        if epoch % 10 == 0:
            self.writer.add_image('real_a', imgs['real_A'].squeeze().cpu(), epoch)
            self.writer.add_image('fake_b', imgs['fake_B'].squeeze().cpu(), epoch)

def main():
    args = get_args()
    get_cfg(args)
    print(cfg)
    trainer = Trainer(cfg)
    tbar = tqdm(range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCH + cfg.TRAIN.ADD_EPOCH))
    for epoch in tbar:
        trainer.train(epoch)
    trainer.writer.close()

if __name__ == '__main__':
    main()
    # print(cfg.TRAIN.START_EPOCH)
    # loss = get_loss_class(cfg, 1)
    # print(loss.__name__)