"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

optimizer learning rate scheduling helpers
"""
from math import ceil


def noam_schedule(step, warmup_step=4000):
    """ original Transformer schedule"""
    if step <= warmup_step:
        return step / warmup_step
    return (warmup_step ** 0.5) * (step ** -0.5)


def warmup_linear(step, warmup_step, tot_step):
    """ BERT schedule """
    if step < warmup_step:
        return step / warmup_step
    return max(0, (tot_step-step)/(tot_step-warmup_step))


def vqa_schedule(step, warmup_interval, decay_interval,
                 decay_start, decay_rate):
    """ VQA schedule from MCAN """
    if step < warmup_interval:
        return 1/4
    elif step < 2 * warmup_interval:
        return 2/4
    elif step < 3 * warmup_interval:
        return 3/4
    elif step >= decay_start:
        num_decay = ceil((step - decay_start) / decay_interval)
        return decay_rate ** num_decay
    else:
        return 1

def get_lr_sched(global_step, sched_type, opts):
    # learning rate scheduling
    if sched_type == 'linear_decay':
        lr_this_step = get_lr_sched_linear_decay(global_step, opts.learning_rate, 
                        opts.warmup_steps, opts.num_train_steps)
    elif sched_type == 'fixed':
        lr_this_step = get_lr_sched_fix(global_step, opts.learning_rate, 
                        opts.warmup_steps)
    else:
        print('[Debug] Error LR sched_type!!! only suppor linear_decay and fixed')
    return lr_this_step

def get_backbone_lr_sched(global_step, sched_type, opts):
    if sched_type == 'linear_decay':
        lr_this_step = get_lr_sched_linear_decay(global_step, opts.backbone_learning_rate, 
                        opts.backbone_warmup_steps, opts.num_train_steps)
    elif sched_type == 'fixed':
        lr_this_step = get_lr_sched_fix(global_step, opts.backbone_learning_rate, 
                        opts.backbone_warmup_steps)
    else:
        print('[Debug] Error LR sched_type!!! only suppor linear_decay and fixed')
    return lr_this_step

def get_lr_sched_linear_decay(global_step, learning_rate, warmup_steps, num_train_steps):
    # learning rate scheduling
    lr_this_step = learning_rate * warmup_linear(
        global_step, warmup_steps, num_train_steps)
    if lr_this_step <= 0:
        lr_this_step = 1e-8
    return lr_this_step

def get_lr_sched_fix(global_step, learning_rate, warmup_steps):
    # learning rate scheduling
    if global_step < warmup_steps:
        lr_this_step = 1e-8
    else:
        lr_this_step = learning_rate
    return lr_this_step