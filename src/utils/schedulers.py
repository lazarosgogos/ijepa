# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math


class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr

class PKTSchedule(object):
    """ The scheduler for the alpha value, to implement PKT
    combined with L2 to enhance learning ability.

    Attributes
    ----------
    warmup_steps : int
        the steps where only PKT should be used
    start_alpha : float
        the value of alpha in the beggining
    ref_alpha : float
        the value of alpha in the intermediate step
    T_max : int
        the epoch alpha reaches its final value
    final_alpha : float
        the final value of alpha
    """
    def __init__(
        self,
        warmup_steps,
        start_alpha,
        ref_alpha,
        T_max,
        final_alpha=0.
    ):
        self.start_alpha = start_alpha
        self.ref_alpha = ref_alpha
        self.final_alpha = final_alpha
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        
        if self._step < self.warmup_steps:
            # progress = float(self._step) / float(max(1, self.warmup_steps))
            # alpha = self.start_alpha + progress * (self.ref_lr - self.start_alpha)
            alpha = 1. # initially, alpha is steadily 1.
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max)) 
            alpha = max(self.final_alpha,
                         self.final_alpha + (self.ref_alpha - self.final_alpha) * 0.5 * (1. + math.cos(math.pi * progress))
                        )

        return alpha


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
