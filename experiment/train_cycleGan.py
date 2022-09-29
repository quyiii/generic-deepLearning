import os
import sys
import pdb
import torch
import argparse
from tqdm import tqdm

sys.path.append('.')
from lib.config import cfg
from lib.util import creat_saver_writer
from lib.solver import get_loss_class
from lib.modeling import create_model
from lib.data import get_dataLoader
from lib.evaluation import get_averageMeter
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
        self.is_best = True
        self.saver, self.writer = creat_saver_writer(cfg)
        self.best = 0.0 if len(cfg.METRIC.NAME) else float('inf')
        self.loss = get_averageMeter()

        self.model = create_model(cfg)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        self.dataloader = get_dataLoader(cfg)
    
    def train(self, epoch):
        # tbar = tqdm(self.dataloader)
        for i, data in enumerate(self.dataloader):
            # print(type(data))
            # print(data)
            # pdb.set_trace()
            # for example loss is tensor(1.) shape is size([]) 
            loss = self.model.optimize_parameters(data)
            # print(type(loss))    float
            self.loss.update(loss)
            if i == 100:
                # fasten the training
                break
        # imgs = self.model.base_model.get_current_visuals()
        # of course torch.Tensor
        # print("img type: ", type(imgs['real_A'][0]))
        self.writer.add_scalar('loss', self.loss.avg, epoch)
        # if epoch == 1 or epoch % 5 == 0:
        #     self.writer.add_image('real_a', imgs['real_A'][0].cpu(), epoch)
        #     self.writer.add_image('fake_b', imgs['fake_B'][0].cpu(), epoch)
        if self.loss.avg < self.best:
            is_best = True
            self.best = self.loss.avg
            self.saver.save_chekpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_perform': 'loss {}'.format(self.loss.avg)
            }, is_best)
        return self.loss.avg

def main():
    args = get_args()
    get_cfg(args)
    print(cfg)
    trainer = Trainer(cfg)
    start_epoch = cfg.TRAIN.START_EPOCH
    total_epoch = cfg.TRAIN.MAX_EPOCH + cfg.TRAIN.ADD_EPOCH
    tbar = tqdm(range(start_epoch, total_epoch))
    for epoch in tbar:
        tbar.set_description('epoch: {}/{}'.format(epoch+1, total_epoch))
        loss = trainer.train(epoch)
        tbar.set_postfix(loss='{}'.format(loss))
    trainer.writer.close()

if __name__ == '__main__':
    main()
    # print(cfg.TRAIN.START_EPOCH)
    # loss = get_loss_class(cfg, 1)
    # print(loss.__name__)