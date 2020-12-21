"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for Emo Recognition Model
"""
from collections import defaultdict
from tqdm import tqdm
import numpy as np

import torch
from horovod import torch as hvd
from torch import nn
from torch.nn import CrossEntropyLoss
from code.uniter.model.model import UniterPreTrainedModel, UniterModel
from code.uniter.utils.misc import NoOp
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from code.uniter.model.layer import GELU
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


class UniterForEmoRecognition(UniterPreTrainedModel):
    """ Finetune UNITER for Emotion Recognition
    """
    def __init__(self, config, img_dim, cls_num, frozen_en_layers):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), 
            GELU(),
            LayerNorm(config.hidden_size, eps=1e-12),
            nn.Dropout(self.config.cls_dropout_prob),
            nn.Linear(config.hidden_size, cls_num)
            )
        self.frozen_en_layers = frozen_en_layers
        self.criterion = CrossEntropyLoss()
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        '''
        if compute_loss is true, the function will return the loss = (1)
        else the the function will return the logits = (batch, cls_num)
        '''
        batch = defaultdict(lambda: None, batch)
        sequence_output = self.uniter(batch, frozen_en_layers=self.frozen_en_layers,
                                      output_all_encoded_layers=False)
        # the output of the first token [CLS]
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.output(pooled_output)
        # one-hot targets
        self.pred = torch.softmax(logits, dim=-1)
        if compute_loss:
            cls_loss = self.criterion(logits, batch['targets'])
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
    total_pred = []
    total_target = []
    eval_loss = 0
    for i, batch in enumerate(loader):
        out = model(batch, compute_loss=False)
        loss = model.criterion(out, batch['targets'])
        eval_loss += loss.item()
        # the predicton reuslts
        preds = model.pred.argmax(dim=1).detach().cpu().numpy()
        targets = batch['targets'].detach().cpu().numpy()
        total_pred.append(preds)
        total_target.append(targets)
        pbar.update(1)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_target)
    avg_loss = eval_loss / len(total_pred)
    try:
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')
        cm = confusion_matrix(total_label, total_pred)
    except:
        acc, uar, f1, cm =0, 0, 0, 0
    model.train()
    pbar.close()
    return {'loss': avg_loss,  'WA': acc,  'UA': uar, 'F1': f1}