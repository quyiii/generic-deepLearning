import os
import torch
import shutil
from collections import OrderedDict
import glob

class Saver(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.directory = os.path.join('experiments', cfg.TASK_NAME, cfg.MODEL.NAME, cfg.DATASET.NAME)
        # get sorted list of files with according directory and name
        self.experiments = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        experiment_id = len(self.experiments) if self.experiments else 0
        
        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(experiment_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
    
    def save_chekpoint(self, state, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['epoch'] = self.cfg.TRAIN.MAX_EPOCH

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()