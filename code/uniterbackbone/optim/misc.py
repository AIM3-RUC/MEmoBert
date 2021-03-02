"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax, SGD

from code.uniterbackbone.optim.adamw import AdamW


def build_optimizer(model, opts, except_model=None):
    '''
    如果有多组参数 和 多个优化器
    paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
    self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
    self.optimizers.append(self.optimizer)
    '''
    if except_model is None:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = [x for x in model.named_parameters() if x not in except_model.named_parameters()]
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

def build_backbone_optimizer(model, opts, except_model=None):
    '''
    如果有多组参数 和 多个优化器
    paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
    self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
    self.optimizers.append(self.optimizer)
    '''
    if except_model is None:
        param_optimizer = list(model.named_parameters())
    else:
        param_optimizer = [x for x in model.named_parameters() if x not in except_model.named_parameters()]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.backbone_weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.backbone_optim == 'adam':
        OptimCls = Adam
    elif opts.backbone_optim == 'adamax':
        OptimCls = Adamax
    elif opts.backbone_optim == 'adamw':
        OptimCls = AdamW
    elif opts.backbone_optim == 'sgd':
        OptimCls = SGD
    else:
        raise ValueError('invalid optimizer')
    if opts.backbone_optim == 'sgd':
        print('[INFO] Use the SGD as backbone optimizer!')
        optimizer = OptimCls(optimizer_grouped_parameters, \
                lr=opts.backbone_learning_rate, \
                momentum=opts.backbone_momentum, \
                weight_decay=opts.backbone_weight_decay, \
                nesterov=opts.backbone_nesterov
            )
    else:
        print('[INFO] Use the {} as backbone optimizer!'.format(opts.backbone_optim))
        optimizer = OptimCls(optimizer_grouped_parameters,
                        lr=opts.backbone_learning_rate, betas=opts.backbone_betas,
                        weight_decay=opts.backbone_weight_decay)
    return optimizer
