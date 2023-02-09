import math
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['WarmUpCosineAnnealingLR', 'FakeLR', 'LearningRateScheduler']


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


class FakeLR(_LRScheduler):
    def __init__(self, optimizer):
        super(FakeLR, self).__init__(optimizer=optimizer)

    def get_lr(self):
        return self.base_lrs


class LearningRateScheduler(_LRScheduler):
    def __init__(self,
                 mode,
                 lr,
                 target_lr=None,
                 num_training_instances=None,
                 stop_epoch=None,
                 warmup_epoch=None,
                 stage_list=None,
                 stage_decay=None,
                 ):
        self.mode = mode
        self.lr = lr
        self.target_lr = target_lr if target_lr is not None else 0
        self.num_training_instances = num_training_instances if num_training_instances is not None else 1
        self.stop_epoch = stop_epoch if stop_epoch is not None else np.inf
        self.warmup_epoch = warmup_epoch if warmup_epoch is not None else 0
        self.stage_list = stage_list if stage_list is not None else None
        self.stage_decay = stage_decay if stage_decay is not None else 0

        self.num_received_training_instances = 0

    def update_lr(self, batch_size):
        self.num_received_training_instances += batch_size

    def get_lr(self, num_received_training_instances=None):
        if num_received_training_instances is None:
            num_received_training_instances = self.num_received_training_instances

        # start_instances = self.num_training_instances * self.start_epoch
        stop_instances = self.num_training_instances * self.stop_epoch
        warmup_instances = self.num_training_instances * self.warmup_epoch

        assert stop_instances > warmup_instances

        current_epoch = self.num_received_training_instances // self.num_training_instances

        if num_received_training_instances < warmup_instances:
            return float(num_received_training_instances + 1) / float(warmup_instances) * self.lr

        ratio_epoch = float(num_received_training_instances - warmup_instances + 1) / \
                      float(stop_instances - warmup_instances)

        if self.mode == 'cosine':
            factor = (1 - np.math.cos(np.math.pi * ratio_epoch)) / 2.0
            return self.lr + (self.target_lr - self.lr) * factor
        elif self.mode == 'stagedecay':
            stage_lr = self.lr
            for stage_epoch in self.stage_list:
                if current_epoch < stage_epoch:
                    return stage_lr
                else:
                    stage_lr *= self.stage_decay
                pass  # end if
            pass  # end for
            return stage_lr
        elif self.mode == 'linear':
            factor = ratio_epoch
            return self.lr + (self.target_lr - self.lr) * factor
        else:
            raise RuntimeError('Unknown learning rate mode: ' + self.mode)
        pass  # end if