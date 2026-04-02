"""
自定义学习率调度器,用于支持 PaddlePaddle 迁移

本模块提供了与 PyTorch SequentialLR 兼容的最小化实现。
"""


class SimpleSequentialScheduler:
    """
    最小化的 SequentialLR 替代品,完全模拟原始行为

    该调度器在指定的里程碑步数切换内部调度器,完全依赖内部调度器的实现。

    参数:
        optimizer: 优化器实例
        schedulers: 调度器列表,例如 [warmup_scheduler, decay_scheduler]
        milestones: 切换点列表,例如 [warmup_steps]
                   当 _step_count >= milestones[i] 时,切换到 schedulers[i+1]
        last_epoch: 用于恢复训练的步数,默认为 -1 (从头开始)

    示例:
        >>> import torch
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        >>>
        >>> # 创建 warmup 调度器
        >>> warmup = torch.optim.lr_scheduler.LinearLR(
        ...     optimizer, start_factor=0.01, end_factor=1.0, total_iters=1000
        ... )
        >>>
        >>> # 创建 decay 调度器
        >>> decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        ...     optimizer, eta_min=0.001, T_max=9000
        ... )
        >>>
        >>> # 组合调度器
        >>> scheduler = SimpleSequentialScheduler(
        ...     optimizer, [warmup, decay], [1000], last_epoch=-1
        ... )
        >>>
        >>> # 训练循环
        >>> for epoch in range(epochs):
        ...     for batch in dataloader:
        ...         optimizer.zero_grad()
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()

    注意:
        - 该调度器完全依赖内部调度器的实现,不重新实现数学公式
        - 不实现 state_dict()/load_state_dict(),完全依赖 last_epoch 机制
        - 可以轻松适配到 PaddlePaddle(只需修改内部调度器的创建方式)
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        """
        初始化调度器切换器

        参数:
            optimizer: 优化器实例
            schedulers: 调度器列表
            milestones: 切换点列表(步数)
            last_epoch: 用于恢复训练的步数,默认为 -1
        """
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self._step_count = 0
        if last_epoch >= 0:
            self._step_count = last_epoch + 1
        scheduler_idx = self._get_scheduler_index()
        current_lr = self.schedulers[scheduler_idx].get_lr()
        self.optimizer.set_lr(current_lr)
        self._last_lr = [current_lr]

    def _get_scheduler_index(self):
        """确定当前应该使用哪个调度器"""
        scheduler_idx = 0
        for i, milestone in enumerate(self.milestones):
            if self._step_count >= milestone:
                scheduler_idx = i + 1
        return scheduler_idx

    def step(self):
        """
        执行一步学习率更新

        根据当前步数确定应该使用哪个调度器,然后调用该调度器的 step() 方法。
        """
        scheduler_idx = self._get_scheduler_index()
        self.schedulers[scheduler_idx].step()
        self._step_count += 1
        current_lr = self.schedulers[scheduler_idx].get_lr()
        self.optimizer.set_lr(current_lr)
        self._last_lr = [current_lr]

    def get_last_lr(self):
        """
        返回当前学习率(用于日志记录)

        返回:
            包含当前学习率的列表
        """
        return self._last_lr
