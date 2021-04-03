import numpy as np
import torch
import json
import logging
from torch import nn
from code.uniter3flow.model.model import BertConfig, BertPreTrainedModel, BertEncoder
from code.uniter.model.layer import BertLayer, BertPooler
from apex.normalization.fused_layer_norm import FusedLayerNorm
from code.uniter3flow.model.enc_text import TextEncoderBertModel
from code.uniter3flow.model.enc_speech import SpeechEncoderBertModel
from code.uniter3flow.model.enc_visual import VisualEncoderBertModel

'''
Cross-encoder model, may including 2~4 transformer layers.
用于接收三个模态的 encoder 的输出，然后做 Concat + attention-mask (音频需要有降采样，所以没有Attention Mask) 进行拼接.
整理的实现参考HERO的实现，HERO的实现也是基于Uniter的框架，所以参考比较方便一些。
f_config 对应 cross-transformer, 6层，但是采用 robota base 中的6层作为初始化, 1 3 5 7 9 11 层的参数进行初始化。type_vocab_size=1.
c_config 对应 temporal-transformer, 3层，type_vocab_size=2.
首先 文本 input_ids, 不需要加 SEP, wordembeeding + positionembedding + typeembedding(1)
https://github.com/linjieli222/HERO/blob/master/model/embed.py#L12
然后 视觉 input_ids, 不需要加SEP, transformed_im + position_embeddings + type_embeddings(1)
https://github.com/linjieli222/HERO/blob/faaf15d6ccc3aa4accd24643d77d75699e9d7fae/model/encoder.py#L247
文本和视觉的type都是1有点奇怪。
然后经过 Cross-Transformer 输入是 [CLS] + input_ids/img_fts + [SEP] + img_fts/input_ids, HERO 采用的是 concat[img_emb,txt_emb]. Img在前面。
在 Cross-Transformer 层上只有一个MLM的预训练任务。输入是 img + txt 来预测 masked token.
最后过 Temporal-Transformer 的输入是: 直接 Cross-Transformer 的输出，如果需要mask/shuffle的话，则是在 Cross-Transformer 之后做。
https://github.com/linjieli222/HERO/blob/f938515424b5f3249fc1d2e7f0373f64112a6529/model/encoder.py#L287

关于视觉和语音模态是否要加 CLS Token, 可以自定义token进行分类。
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py#L88
'''

class CrossEncoderBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    Input: 
        text embeddings from text encoder
        text_att_mask embeddings from text encoder
        audio embeddings from text encoder
        audio_att_mask embeddings from audio encoder
        visual embeddings from visal encoder
        visual_att_mask embeddings from visal encoder
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = TextEncoderBertModel(config.t_config)
        self.visual_encoder = VisualEncoderBertModel(config.v_config)
        self.speech_encoder = SpeechEncoderBertModel(config.s_config)

        self.cross_encoder = BertEncoder(config.c_config)
        # due to the viseme is 512 and trans to hidden_size
        self.apply(self.init_weights)

    def forward(self, text_input_batch, visual_input_batch, speech_input_batch, output_all_encoded_layers=False):
        print(f'[Debug] text_input_batch {text_input_batch.shape}')
        extended_text_att_mask = text_input_batch['extended_att_mask']
        text_encoder_output = self.text_encoder(text_input_batch, extended_text_att_mask)
        print(f'[Debug] text_encoder_output {text_encoder_output.shape}')
        print(f'[Debug] visual_input_batch {visual_input_batch.shape}')
        extended_visual_att_mask = visual_input_batch['extended_att_mask']
        visual_encoder_output = self.visual_encoder(visual_input_batch, extended_visual_att_mask) 
        print(f'[Debug] visual_encoder_output {visual_encoder_output.shape}')
        print(f'[Debug] speech_input_batch {speech_input_batch.shape}')
        extended_speech_att_mask = speech_input_batch['extended_att_mask']
        speech_encoder_output = self.visual_encoder(speech_input_batch, extended_speech_att_mask) 
        print(f'[Debug] speech_encoder_output {speech_encoder_output.shape}')

        # combine the three modality output and attention mask (batch, seq-len, dim)
        combine_modality_output = torch.cat([text_encoder_output, speech_encoder_output, visual_encoder_output], dim=1)
        combine_modality_attention_mask = torch.cat([extended_text_att_mask, extended_speech_att_mask, extended_visual_att_mask], dim=1)
        print(f'[Debug] combined modality output {combine_modality_output.shape} and combine_modality_attention_mask {combine_modality_attention_mask.shape}')
        encoded_layers = self.encoder(
            combine_modality_output, combine_modality_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

if __name__ == '__main__':
    main()
    
