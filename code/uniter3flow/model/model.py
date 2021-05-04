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
from code.uniter3flow.model.model_base import BertConfig, BertPreTrainedModel
from code.uniter3flow.model.enc_speech_new import SpeechWav2Vec2Model
from transformers import Wav2Vec2Config
from code.uniter3flow.model.enc_visual_new import VisualEncoderBertModel
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
    def __init__(self, config, use_speech, use_visual,
                                pretrained_text_checkpoint=None,
                                pretrained_audio_checkpoint=None,
                                fix_text_encoder=False,
                                fix_visual_encoder=False,
                                fix_speech_encoder=False,
                                fix_cross_encoder=False,
                                use_type_embedding=False):
        super().__init__(config)
        self.c_config = BertConfig.from_dict(config.c_config)  # for cross transformer 
        self.v_config = BertConfig.from_dict(config.v_config)  # for visual transformer 
        self.t_config = BertConfig.from_dict(config.t_config)  # for text transformer 
        self.s_config = Wav2Vec2Config.from_dict(config.s_config)  # for speech transformer
        self.use_speech = use_speech
        self.use_visual = use_visual
        self.fix_text_encoder = fix_text_encoder
        self.fix_visual_encoder = fix_visual_encoder
        self.fix_speech_encoder = fix_speech_encoder
        self.fix_cross_encoder = fix_cross_encoder
        self.use_type_embedding = use_type_embedding

        self.text_encoder = TextEncoderBertModel(self.t_config)
        # text encoder need pretraind model
        if pretrained_text_checkpoint is None:
            logger.info('[INFO] Scratch the text encoder')
            checkpoint = {}
        else:
            logger.info('[INFO] Loading the text pretrained model {}'.format(pretrained_text_checkpoint))
            checkpoint = torch.load(pretrained_text_checkpoint)
        self.text_encoder = self.text_encoder.from_pretrained(
                    self.t_config, checkpoint)

        if use_speech:
            logger.info('[Info] use the wav2vec speech encoder')
            self.speech_encoder = SpeechWav2Vec2Model(self.s_config)
            if pretrained_audio_checkpoint:
                logger.info('[INFO] Loading the wav2vec pretrained model {}'.format(pretrained_audio_checkpoint))
                self.speech_encoder.encoder.load_state_dict(torch.load(pretrained_audio_checkpoint))
            else:
                logger.info('[INFO] Scratch the speech wav2vec2 encoder')

        if use_visual:
            logger.info('[Info] use the visual branch')
            self.visual_encoder = VisualEncoderBertModel(self.v_config)

        self.cross_encoder = CrossEncoderBertModel(self.c_config)
        if self.use_type_embedding:
            logger.info('[Info] use the type embedding for each modality')
            self.token_type_embeddings = nn.Embedding(self.c_config.type_vocab_size,
                                                    self.c_config.hidden_size)
             
        self.apply(self.init_weights)

    def _compute_modalities_embeddings(self, combine_modality_outputs, gather_index):
        # combine_modality_outputs may including two or three modalities
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat(combine_modality_outputs, dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, output_all_encoded_layers=False):
        combine_modality_outputs = []
        combine_modality_att_masks = []
        # case1: use text
        if self.fix_text_encoder:
            with torch.no_grad():
                text_encoder_output, text_extended_att_mask = self.text_encoder(batch)
        else:
            text_encoder_output, text_extended_att_mask = self.text_encoder(batch)

        if self.use_type_embedding:
            # add text type embeddings, using index 0
            text_token_type_ids = torch.zeros_like(batch['input_ids'])
            token_type_embeddings = self.token_type_embeddings(text_token_type_ids)
            text_encoder_output = text_encoder_output + token_type_embeddings
        # print(f'[Debug] text_encoder_output {text_encoder_output.shape}')
        # print(f'[Debug] text_extended_att_mask {text_extended_att_mask.shape}')
        combine_modality_outputs.append(text_encoder_output)
        combine_modality_att_masks.append(text_extended_att_mask)

        if self.use_visual:
            if self.fix_visual_encoder:
                 with torch.no_grad():
                    visual_encoder_output, visual_extended_att_mask = self.visual_encoder(batch, output_all_encoded_layers=False) 
            else:
                visual_encoder_output, visual_extended_att_mask = self.visual_encoder(batch, output_all_encoded_layers=False)
            if self.use_type_embedding:
                # add visual type embeddings, use index 1
                visual_token_type_ids = torch.ones_like(visual_encoder_output[:, :, 0].long())
                visual_token_type_embeddings = self.token_type_embeddings(visual_token_type_ids)
                visual_encoder_output = visual_encoder_output + visual_token_type_embeddings
            combine_modality_outputs.append(visual_encoder_output)
            combine_modality_att_masks.append(visual_extended_att_mask)
            # print(f'[Debug] visual_encoder_output {visual_encoder_output.shape}')
            # print(f'[Debug] visual_extended_att_mask {visual_extended_att_mask.shape}')

        if self.use_speech:
            if self.fix_speech_encoder:
                 with torch.no_grad():
                    speech_encoder_output = self.speech_encoder(batch)
            else:
                speech_encoder_output = self.speech_encoder(batch)
            
            speech_attn_mask = batch['speech_attn_masks']
            # compute self-attention mask.
            speech_extended_attn_mask = speech_attn_mask.unsqueeze(1).unsqueeze(2)
            speech_extended_attn_mask = speech_extended_attn_mask.to(
                                        dtype=next(self.parameters()).dtype)  # fp16 compatibility
            speech_extended_attn_mask = (1.0 - speech_extended_attn_mask) * -10000.0  ## torch.Size([16, 1, 1, 297])
            ## for postprocessing
            if speech_encoder_output.size(1) > speech_extended_attn_mask.size(3):
                speech_encoder_output = speech_encoder_output[:, :speech_extended_attn_mask.size(3), :] 
            
            if self.use_type_embedding:
                # add speech type embeddings, also use index 1
                speech_token_type_ids = torch.ones_like(speech_encoder_output[:, :, 0].long()) + \
                                                    torch.ones_like(speech_encoder_output[:, :, 0].long())
                speech_token_type_embeddings = self.token_type_embeddings(speech_token_type_ids)
                speech_encoder_output = speech_encoder_output + speech_token_type_embeddings

            # combine modality info
            combine_modality_outputs.append(speech_encoder_output)
            combine_modality_att_masks.append(speech_extended_attn_mask)
            # print(f'[Debug] A speech_encoder_output {speech_encoder_output.shape}')
            # print(f'[Debug] speech_extended_att_mask {speech_extended_attn_mask.shape}')

        gather_index = batch['gather_index']
        # print('[Debug] input_ids {}'.format(batch['input_ids'][0]))
        # print(f'[Debug] gather_index {gather_index[0]}')
        combine_modality_output = self._compute_modalities_embeddings(combine_modality_outputs, gather_index)
        combine_modality_attention_mask = torch.cat(combine_modality_att_masks, dim=-1)
        # print(f'[Debug] combine_modality_output {combine_modality_output.shape}')
        # print(f'[Debug] combine_modality_attention_mask {combine_modality_attention_mask.shape}')
        # print(f'[Debug] {combine_modality_output[0]}')
        
        if self.fix_cross_encoder:
            with no_grad():
                encoded_layers = self.cross_encoder(
                    combine_modality_output, combine_modality_attention_mask,
                    output_all_encoded_layers=output_all_encoded_layers)
        else:
            encoded_layers = self.cross_encoder(
                    combine_modality_output, combine_modality_attention_mask,
                    output_all_encoded_layers=output_all_encoded_layers)
        # print('[Debug in model] final output {}'.format(encoded_layers.shape))
        return encoded_layers