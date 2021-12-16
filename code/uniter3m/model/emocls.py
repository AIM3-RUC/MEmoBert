"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for Emo Recognition Model
"""
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import collections

import torch
from horovod import torch as hvd
from torch import nn
from torch.nn import CrossEntropyLoss
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from code.uniter3m.model.model import UniterPreTrainedModel, UniterModel
from code.uniter3m.model.pretrain import EmoClassification
## from uniter
from code.uniter.model.layer import GELU
from code.uniter.utils.misc import NoOp

class UniterForEmoRecognition(UniterPreTrainedModel):
    """ Finetune UNITER for Emotion Recognition
    """
    def __init__(self, config, img_dim, speech_dim, cls_num, frozen_en_layers, \
                        use_visual, use_speech, \
                        cls_dropout=0.1, cls_type='vqa', use_emolare=False):
        '''
        cls_type: "emocls" is similar with  https://github.com/brightmart/roberta_zh/blob/master/run_classifier.py#L478
        and "vqa" is similar with official-uniter/model/vqa.py 
        '''
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, speech_dim, use_visual, use_speech)
        ## for paraphrase loss
        self.output = EmoClassification(config.hidden_size, cls_num, cls_dropout=cls_dropout, cls_type=cls_type)
        self.use_emolare = use_emolare
        self.frozen_en_layers = frozen_en_layers
        self.criterion = CrossEntropyLoss()
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        '''
        if compute_loss is true, the function will return the loss = (1)
        else the the function will return the logits = (batch, cls_num)
        '''
        batch = defaultdict(lambda: None, batch)
        sequence_output = self.uniter(batch, use_emolare_input=self.use_emolare, frozen_en_layers=self.frozen_en_layers,
                                      output_all_encoded_layers=False)
        # the output of the first token [CLS] / first token
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.output(pooled_output)
        # one-hot targets
        self.pred = torch.softmax(logits, dim=-1)
        if compute_loss:
            cls_loss = self.criterion(logits, batch['targets'])
            return cls_loss
        else:
            return logits

class UniterForEmoRecognitionPrompt(UniterPreTrainedModel):
    """ Finetune UNITER for Emotion Recognition based Prompt Method
    """
    def __init__(self, config, img_dim, speech_dim, cls_num, frozen_en_layers, \
                        use_visual, use_speech, use_emolare=False):
        '''
        cls_type: "emocls" is similar with  https://github.com/brightmart/roberta_zh/blob/master/run_classifier.py#L478
        and "vqa" is similar with official-uniter/model/vqa.py 
        '''
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, speech_dim, use_visual, use_speech)
        ## for paraphrase loss
        self.use_emolare = use_emolare
        self.frozen_en_layers = frozen_en_layers
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        '''
        if compute_loss is true, the function will return the loss = (1)
        else the the function will return the logits = (batch, cls_num)
        '''
        batch = defaultdict(lambda: None, batch)
        sequence_output = self.uniter(batch, use_emolare_input=self.use_emolare, frozen_en_layers=self.frozen_en_layers,
                                      output_all_encoded_layers=False)
        input_ids = batch['input_ids']
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)
        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                            txt_labels[txt_labels != -1],
                                            reduction='none')
            # print('[Debug] in MLM and lmloss {}'.format(masked_lm_loss.shape)) # all validate token loss torch.Size([35])
            return masked_lm_loss
        return prediction_scores
       

@torch.no_grad()
def evaluation(model, loader):
    model.eval()
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
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_target)
    avg_loss = eval_loss / len(total_pred)
    try:
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        wf1 = f1_score(total_label, total_pred, average='weighted')
        f1 = f1_score(total_label, total_pred, average='macro')
        cm = confusion_matrix(total_label, total_pred)
    except:
        acc, uar, wf1, f1, cm =0, 0, 0, 0,0
    model.train()
    return {'loss': avg_loss, 'WA': acc, 'WF1': wf1, 'UA': uar,  'F1': f1}

@torch.no_grad()
def evaluation_miss_conditions(model, val_dataloaders):
    # 统计6种不同模态缺失场景的结果选择模型, 平均值即可, 整理为如下的结果
    # task2logs = {'testlva': {'WA', 'UA'}, 'testlv':[], testla:[], testl:[],
    # 'testva': [], 'testlv':[], testla:[], testl:[], 'miss6coditions': []}
    model.eval()
    task2logs = collections.OrderedDict()
    for name in val_dataloaders.keys():
        condition_name = name.split('_')[1]
        loader = val_dataloaders[name]
        val_log = evaluation(model, loader)
        task2logs[condition_name] = val_log
    model.train()
    # average six_missing conditions results
    miss_mean_wa, miss_mean_uar = 0, 0
    for c_name in task2logs.keys():
        if 'lva' not in c_name:
            temp_dict = task2logs[c_name]
            miss_mean_wa += temp_dict['WA']
            miss_mean_uar += temp_dict['UA']
    miss_mean_wa = miss_mean_wa/6
    miss_mean_uar = miss_mean_uar/6
    task2logs['miss6coditions'] = {'WA': miss_mean_wa, 'UA':miss_mean_uar}
    return task2logs

@torch.no_grad()
def evaluation_prompt(model, val_loader):
    # 计算 wa, uar, f1
    LOGGER.info(f"start running prompt-based evaluation...")
    val_loss = 0
    total_preds = []
    total_labels = []
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task=task, compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        temp_preds = scores.argmax(axis=1)
        # print(temp_preds.shape, labels.shape)
        total_preds.append(temp_preds.detach().cpu().numpy())
        total_labels.append(labels.detach().cpu().numpy())
    val_loss = sum(all_gather_list(val_loss))
    tot_time = time()-st
    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)   
    val_loss /= len(total_labels)
    # print(total_preds.shape, total_labels.shape)
    val_log = evaluation_metric(total_preds, total_labels)
    val_log['loss'] = val_loss
    return val_log

def evaluation_metric(total_pred, total_label):
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    wf1 = f1_score(total_label, total_pred, average='weighted')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    return {'WA': acc, 'WF1': wf1, 'UA': uar,  'F1': f1}