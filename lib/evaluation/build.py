from .meter import AverageMeter

def get_averageMeter():
    meter = AverageMeter()
    meter.reset()
    return meter