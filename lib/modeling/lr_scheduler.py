
def get_scheduler(optimizer, cfg):
    if cfg.SOLVER.LR_SCHEDULER == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, )