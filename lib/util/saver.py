import os
import torch
import shutil
from collections import OrderedDict
import glob

class Saver(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.directory = os.path.join('run', cfg.TASK_NAME, cfg.MODEL.NAME, cfg.DATASET.NAME)
        # get sorted list of files with according directory and name
        self.experiments = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        if self.experiments:
            last_experiment = self.experiments[-1]
            self.experiment_id = self.get_experiment_id(last_experiment) + 1
        else:
            self.experiment_id = 0
        
        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(self.experiment_id))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
    
    def get_experiment_id(self, experiment_path):
        pos = experiment_path.rfind('_')
        if pos == -1:
            raise RuntimeError("experiment {} need number".format(experiment_path))
        else:
            return int(experiment_path[pos+1:])

    def get_best_perform_val(self, best_perform):
        pos = best_perform.rfind(' ')
        if pos == -1:
            raise RuntimeError("best_perform {} need space".format(best_perform))
        else:
            return float(best_perform[pos+1:])

    def save_best(self, best_perform, checkpoint, best_name):
        with open(os.path.join(self.directory, 'best_experiment.txt'), 'w') as f:
            f.write("experiment{}\n".format(self.experiment_id))
            f.write(best_perform)
        shutil.copyfile(checkpoint, best_name)

    def save_chekpoint(self, state, is_best=True, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            # best_perform: loss 1.0 or acc 1.0 ...
            best_perform_val = self.get_best_perform_val(state['best_perform'])
            with open(os.path.join(self.experiment_dir, 'best_perform.txt'), 'w') as f:
                f.write(state['best_perform'])
            model_best_name = os.path.join(self.directory, "mdoel_best.pth.tar")
            if self.experiments:
                is_big_better = True if len(self.cfg.METRIC.NAME) else False
                previous_vals = [0.0] if is_big_better else [float('inf')]
                for experiment in self.experiments:
                    best_path = os.path.join(experiment, 'best_perform.txt')
                    if os.path.exists(best_path):
                        with open(best_path, 'r') as f:
                            previous_vals.append(self.get_best_perform_val(f.readline()))
                if is_big_better:
                    previous_best = max(previous_vals)
                    if best_perform_val > previous_best:
                        self.save_best(state["best_perform"], filename, model_best_name)
                else:
                    previous_best = min(previous_vals)
                    if best_perform_val < previous_best:
                        self.save_best(state["best_perform"], filename, model_best_name)
            else:
                self.save_best(state["best_perform"], filename, model_best_name)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'config.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['epoch'] = self.cfg.TRAIN.MAX_EPOCH

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()