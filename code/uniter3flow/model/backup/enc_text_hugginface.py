import numpy as np
import torch
import json
import logging
from torch import nn
from transformers import BertModel, BertPreTrainedModel, BertConfig

''' 
实现思路:
1. 首先文本输入正常加载预训练的模型, 采用 huggingface 的
https://huggingface.co/transformers/main_classes/model.html
https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
里面有各个不同模块的实现
''' 

logger = logging.getLogger(__name__)

class TextBertEncoder(BertPreTrainedModel):
    def __init__(self, config, checkpoint=None, bert_type='bert-base-uncased', embd_method='cls'):
        super().__init__(config)
        self.embd_method = embd_method
        if self.embd_method not in ['cls', 'mean', 'max']:
            raise NotImplementedError('Only [cls, mean, max] embd_method is supported, \
                but got', config.embd_method)
        
        if bert_type == 'bert-base-uncased':
            print('[Info] Use the official bert-base-uncased pretrained model')
            self.bert = BertModel.from_pretrained(bert_type)
        else:
            if checkpoint is not None:
                print('[Info] Use the opensub1000w pretrained bert-base-uncased model')
                self.bert = BertModel.from_pretrained(checkpoint)
            else:
                logger.info('[Warning] From scratch')
                self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask):
        '''
        Feed the input to Bert model to obtain contextualized representations
        return: hidden_states with shape (number_layers, bs, seq_len, 768)
        return: cls_reps (number_layers, bs, seq_len, 768)
        '''
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            position_ids=None,
            token_type_ids=None
        )
        last_hidden = outputs.last_hidden_state
        cls_token = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # using different embed method
        if self.embd_method == 'cls':
            cls_reps = cls_token
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]
        print('hidden_states {}'.format(hidden_states.shape))
        print('cls_reps {}'.format(cls_reps.shape))
        return hidden_states, cls_reps

if __name__ == '__main__':
    # pretrained_checkpoint = '/data7/emobert/resources/pretrained/uniter-base-uncased-init.pt'
    pretrained_checkpoint = '/data7/MEmoBert/emobert/exp/mlm_pretrain/results/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980'
    config_path = '/data7/MEmoBert/code/uniter3flow/config/uniter-base.json'
    config = json.load(open(config_path, 'r'))
    config = BertConfig().from_dict(config)
    TextBertEncoder(config, bert_type='bert-base-uncased')