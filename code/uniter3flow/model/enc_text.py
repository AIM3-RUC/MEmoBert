import numpy as np
import torch
import json
import logging
from torch import nn
from code.uniter3flow.model.model import BertConfig, BertPreTrainedModel, BertEncoder
from code.uniter.model.layer import BertLayer, BertPooler
from apex.normalization.fused_layer_norm import FusedLayerNorm

''' 
实现思路:
1. 首先文本输入正常加载预训练的模型, 采用 huggingface 的
https://huggingface.co/transformers/main_classes/model.html
https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertModel
里面有各个不同模块的实现
''' 

logger = logging.getLogger(__name__)

class BertTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config: 
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        # build position vocab embeddings = 512
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None, use_token_type=False):
        '''
        emo_type_ids: the emotion types of the input ids
        batch-data
        '''
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        
        if use_token_type:
            logger.info('[Info] Use the token type embeddings')
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
  
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TextEncoderBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertTextEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def forward(self, batch, output_all_encoded_layers=False, txt_type_ids=None, use_token_type=False):
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        attention_mask = batch['attn_masks']
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, position_ids, txt_type_ids, use_token_type)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

if __name__ == '__main__':
    pretrained_checkpoint = '/data7/emobert/resources/pretrained/uniter-base-uncased-init.pt'
    # pretrained_checkpoint = '/data7/MEmoBert/emobert/exp/mlm_pretrain/results/opensub/bert_base_uncased_1000w_linear_lr1e4_warm4k_bs256_acc2_4gpu/checkpoint-93980'
    config_path = '/data7/MEmoBert/code/uniter3flow/config/uniter-base.json'
    config = BertConfig(config_path)
    checkpoint = torch.load(pretrained_checkpoint)
    txt_encoder = TextEncoderBertModel(config)
    txt_encoder.from_pretrained(config_path, checkpoint)
    input_batch = {
        'input_ids': torch.tensor([[101, 1111, 1113, 102]]),
        'position_ids': torch.tensor([[0, 0, 0, 0]]),
        'attn_masks': torch.tensor([[1, 1, 1, 1]])
    }
    encoded_layers = txt_encoder.forward(input_batch)
    print(encoded_layers.shape) # torch.Size([1, 4, 768])