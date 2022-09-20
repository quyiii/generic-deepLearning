from torch.optim import lr_scheduler

def get_scheduler(optimizer, cfg):
    lr_policy = cfg.SOLVER.LR_SCHEDULER.lower()
    if lr_policy == 'linear':
        # 线性调节学习率: lr = lambda * lr
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + cfg.TRAIN.START_EPOCH - cfg.SOLVER.LR_INIT_EPOCH)/float(cfg.SOLVER.LR_DECAY_EPOCH + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        # 等间隔lr_decay_iters调整学习率: lr = lr * gamma 
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.LR_DECAY_ITERS, gamma=0.1)
    elif lr_policy == 'plateau':
        # 在lr达到停滞期时调整学习率: lr = factor * lr 
        # mode 模式选择 min以loss为指标 max以acc为指标
        # patience: 能够忍受多少个epoch的指标不变好 忍无可忍时调整学习率
        # thresholda: 衡量指标是否变好的阈值
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        raise NotImplementedError('lr_scheduler {} is not implemented'.format(lr_policy))
    return scheduler

