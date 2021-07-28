"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from code.uniter3m.optim.adamw import AdamW


def build_optimizer(model, opts):
    '''
    如果有多组参数 和 多个优化器
    paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
    self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
    self.optimizers.append(self.optimizer)
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer
