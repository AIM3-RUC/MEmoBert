"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for Emo Recognition Model
"""
from collections import defaultdict
from tqdm import tqdm

import torch
from horovod import torch as hvd
from torch import nn
from code.uniter.model.model import UniterPreTrainedModel, UniterModel
from code.uniter.utils.misc import NoOp


class UniterForEmoRecognition(UniterPreTrainedModel):
    """ Finetune UNITER for Emotion Recognition
    """
    def __init__(self, config, img_dim, training):
        super().__init__(config)
        self.traning = training
        self.uniter = UniterModel(config, img_dim)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), 
            nn.ReLU(True),
            nn.Linear(config.hidden_size, config.cls_num)
            )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        sequence_output = self.uniter(batch, frozen_en_layers=self.config.frozen_en_layers,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.output(pooled_output)
        # one-hot targets
        targets = batch['targets']
        if compute_loss:
            # softmax loss
            cls_loss = self.criterion(logits, targets)
            return cls_loss
        else:
            return logits

@torch.no_grad()
def evaluation(model, loader):
    model.eval()
    if hvd.rank() == 0:
        pbar = tqdm(total=len(loader))
    else:
        pbar = NoOp()
    predictions_outputs = []
    eval_loss = 0
    eval_acc = 0
    for i, batch in enumerate(loader):
        out = model(batch)
        loss = model.loss_func(out, batch['targets'])
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch['targets']).sum()
        eval_acc += num_correct.item()
        pbar.update(1)
    # average loss and num_correct / total samples in set
    print('eval Loss: {:.6f}, Acc: {:.6f}'.format()) 
    model.train()
    pbar.close()
    return predictions_outputs