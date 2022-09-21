import os
import torch
from tensorboardX import SummaryWriter

class TensorboardSummary(object):
    """ build a online log to save train details
    
    writer = SummaryWriter(file_path)
    writer.add_scalar('title', value, epoch)
    writer.add_image('title', iamge_tensor(3, H, W), epoch)
    writer.close()

    teminal input:
        tensorboard --logdir=.....file_path...
    http://localhost:6006/
    """
    def __init__(self, directory):
        self.directory = directory
    
    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer