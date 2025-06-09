import paddle


def instantiate_scheduler(optimizer, config):
    if config.opt_scheduler == 'CosineAnnealingLR':
        tmp_lr = paddle.optimizer.lr.CosineAnnealingDecay(T_max=config.
            opt_scheduler_T_max, learning_rate=optimizer.get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    elif config.opt_scheduler == 'StepLR':
        tmp_lr = paddle.optimizer.lr.StepDecay(step_size=config.
            opt_step_size, gamma=config.opt_gamma, learning_rate=optimizer.
            get_lr())
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
    else:
        raise ValueError(f'Got config.opt.scheduler={config.opt.scheduler!r}')
    return scheduler
