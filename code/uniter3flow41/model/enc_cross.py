import numpy as np
import torch
import json
import logging
from torch import nn
from code.uniter3flow.model.model_base import BertPreTrainedModel, BertEncoder
from code.uniter3flow.model.layer import BertPooler
from apex.normalization.fused_layer_norm import FusedLayerNorm

class CrossEncoderBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    Input: 
        combine_modality_output, (num_modalities, batch, multi-modality-len, dim)
        combine_modality_attention_mask, (num_modalities, batch, 1, 1 multi-modality-len)
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm. 
    """
    def __init__(self, config):
        super().__init__(config)
        self.cross_encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def forward(self, combine_modality_output, combine_modality_attention_mask, 
                                    output_all_encoded_layers=False):
        '''
        combine_modality_output: (batchsize, len1+len2, dim)
        combine_modality_attention_mask: (batchsize, 1, 1, len1+len2)
        '''
        encoded_layers = self.cross_encoder(
            combine_modality_output, combine_modality_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        # print('[Debug in cross-encoder] one encoded_layer {}'.format(encoded_layers[0].shape))
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers