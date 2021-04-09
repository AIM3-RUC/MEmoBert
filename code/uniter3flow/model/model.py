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
from code.uniter3flow.model.model_base import BertConfig, BertPreTrainedModel
from code.uniter3flow.model.enc_speech import SpeechEncoderBertModel
from code.uniter3flow.model.enc_visual_new import VisualEncoderBertModel
from code.uniter3flow.model.enc_text import TextEncoderBertModel
from code.uniter3flow.model.enc_cross import CrossEncoderBertModel

logger = logging.getLogger(__name__)


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    '''
    gather-index, 为了减少对齐成本，指示真实数据所在的index, 比如这里每条数据, 
    txt=13, img=10, 目前 txt-max=20, img-max=20, pad之后的数据是 
    [txt20, img20], 而进行gather之后的操作是 [txt-13, img10, img[3~20]]
    而 Attention-Mask 本身就是 [txt-13, img10, other-1] 
    因为这里的 Attention-Mask 是先pad后拼接的, 所以应该 gather-index, attention-mask, 
    但是需要注意的是，最后的剩余的部分全是-1.
    '''
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index

class MEmoBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config, use_speech, use_visual, pretrained_text_checkpoint=None):
        super().__init__(config)
        self.c_config = BertConfig.from_dict(config.c_config)  # for cross transformer 
        self.v_config = BertConfig.from_dict(config.v_config)  # for visual transformer 
        self.t_config = BertConfig.from_dict(config.t_config)  # for text transformer 
        self.s_config = BertConfig.from_dict(config.s_config)  # for speech transformer
        self.use_speech = use_speech
        self.use_visual = use_visual
    
        self.text_encoder = TextEncoderBertModel(self.t_config)
        # text encoder need pretraind model
        if pretrained_text_checkpoint is None:
            checkpoint = {}
        else:
            logger.info('[INFO] Loading the pretrained model {}'.format(pretrained_text_checkpoint))
            checkpoint = torch.load(pretrained_text_checkpoint)
        self.text_encoder = self.text_encoder.from_pretrained(
                    self.t_config, checkpoint)
        
        if use_visual:
            logger.info('[Info] use the visual branch')
            self.visual_encoder = VisualEncoderBertModel(self.v_config)
        if use_speech:
            logger.info('[Info] use the speech branch')
            self.speech_encoder = SpeechEncoderBertModel(self.s_config)
        self.cross_encoder = CrossEncoderBertModel(self.c_config)

        self.do_gather  = False
        self.apply(self.init_weights)

    def _compute_img_txt_embeddings(self, txt_emb, img_emb, txt_emb_attn, img_emb_attn, txt_lens, num_bbs):
        '''
        For gather valid info to the front of the sequence. 
        Attention Mask 也得做相应的处理.
        '''
        out_size = txt_emb.size(1) + img_emb.size(1)
        bs, max_tl = txt_emb.size(0), txt_emb.size(1)
        gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.c_config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        embedding_output_attn_masks = torch.gather(torch.cat([txt_emb_attn, img_emb_attn], dim=1),
                                        dim=1, index=gather_index)
        # 需要将 embedding_output_attn_masks 中大于本身长度的部分都置为-1.
        # Pending to implement
        return embedding_output, embedding_output_attn_masks

    def forward(self, batch, output_all_encoded_layers=False):
        combine_modality_outputs = []
        combine_modality_att_masks = []
        # case1: use text
        text_encoder_output, text_extended_att_mask = self.text_encoder(batch)
        # print(f'[Debug] text_encoder_output {text_encoder_output.shape}')
        # print(f'[Debug] text_extended_att_mask {text_extended_att_mask.shape}')
        combine_modality_outputs.append(text_encoder_output)
        combine_modality_att_masks.append(text_extended_att_mask)

        if self.use_visual:
            visual_encoder_output, visual_extended_att_mask = self.visual_encoder(batch, output_all_encoded_layers=False) 
            combine_modality_outputs.append(visual_encoder_output)
            combine_modality_att_masks.append(visual_extended_att_mask)
            # print(f'[Debug] visual_encoder_output {visual_encoder_output.shape}')
            # print(f'[Debug] visual_extended_att_mask {visual_extended_att_mask.shape}')

        if self.use_speech:
            speech_encoder_output, speech_extended_att_mask = self.speech_encoder(batch)
            combine_modality_outputs.append(speech_encoder_output)
            combine_modality_att_masks.append(speech_extended_att_mask)
            # print(f'[Debug] speech_encoder_output {speech_encoder_output.shape}')
            # print(f'[Debug] speech_extended_att_mask {speech_extended_att_mask.shape}')

        if self.do_gather:
            combine_modality_output, combine_modality_attention_mask = self._compute_img_txt_embeddings(combine_modality_outputs[0])
        else:
            combine_modality_output = torch.cat(combine_modality_outputs, dim=1)
            combine_modality_attention_mask = torch.cat(combine_modality_att_masks, dim=-1)

        # print(f'[Debug] combine_modality_output {combine_modality_output.shape}')
        # print(f'[Debug] combine_modality_attention_mask {combine_modality_attention_mask.shape}')
        encoded_layers = self.cross_encoder(
            combine_modality_output, combine_modality_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        # print('[Debug in model] final output {}'.format(encoded_layers.shape))
        return encoded_layers