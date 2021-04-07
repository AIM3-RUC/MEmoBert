"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open

import numpy as np
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from code.uniter.model.layer import BertLayer, BertPooler
from code.uniter3flow.model.model_base import BertPreTrainedModel
from code.uniter3flow.model.enc_speech import SpeechEncoderBertModel
from code.uniter3flow.model.enc_visual import VisualEncoderBertModel
from code.uniter3flow.model.enc_text import TextEncoderBertModel
from code.uniter3flow.model.enc_cross import CrossEncoderBertModel

logger = logging.getLogger(__name__)

class MEmoBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config, use_speech, use_visual):
        super().__init__(config)
        self.c_config = config.c_config  # for cross transformer 
        self.v_config = config.v_config  # for visual transformer 
        self.t_config = config.t_config  # for text transformer 
        self.s_config = config.s_config  # for speech transformer
        self.use_speech = use_speech
        self.use_visual = use_visual
        
        self.text_encoder = TextEncoderBertModel(self.t_config)
        if use_speech:
            logger.info('[Info] use the speech branch')
            self.visual_encoder = VisualEncoderBertModel(self.v_config)
        if use_visual:
            logger.info('[Info] use the visual branch')
            self.speech_encoder = SpeechEncoderBertModel(self.s_config)
        self.cross_encoder = CrossEncoderBertModel(self.c_config)

        self.apply(self.init_weights)


    def _compute_img_txt_embeddings(self, txt_emb, img_emb,
                                    gather_index):
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, frozen_en_layers=0, output_all_encoded_layers=True):
        '''
        关于gather-index, 是text-pad到最大长度的，
        '''
        gather_index = batch['gather_index']

        combine_modality_outputs = []
        combine_modality_att_masks = []
        # case1: use text
        text_encoder_output, text_extended_att_mask = self.text_encoder(batch)
        print(f'[Debug] text_encoder_output {text_encoder_output.shape}')
        combine_modality_outputs.append(text_encoder_output)
        combine_modality_att_masks.append(text_extended_att_mask)

        if self.use_visual:
            visual_encoder_output, visual_extended_att_mask = self.visual_encoder(batch, output_all_encoded_layers=False) 
            combine_modality_outputs.append(visual_encoder_output)
            combine_modality_att_masks.append(visual_extended_att_mask)
            print(f'[Debug] visual_encoder_output {visual_encoder_output.shape}')

        if self.use_speech:
            speech_encoder_output, speech_extended_att_mask = self.speech_encoder(batch)
            combine_modality_outputs.append(speech_encoder_output)
            combine_modality_att_masks.append(speech_extended_att_mask)
            print(f'[Debug] speech_encoder_output {speech_encoder_output.shape}')

        # combine the three modality output and attention mask (batch, seq-len, dim)
        combine_modality_output = torch.cat(combine_modality_outputs, dim=1)
        combine_modality_attention_mask = torch.cat(combine_modality_att_masks, dim=1)

        # filter the 
        combine_modality_output = self._compute_img_txt_embeddings(combine_modality_output, gather_index)

        encoded_layers = self.encoder(
            combine_modality_output, combine_modality_attention_mask,
            frozen_en_layers=frozen_en_layers,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers