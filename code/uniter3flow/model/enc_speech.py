"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
Conv3D + Resnet  from:  https://github.com/lordmartian/deep_avsr
Transformers from: Add by Jinming
visual branch is Conv3D + Resnet + Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from code.uniter3flow.model.model import BertConfig, BertPreTrainedModel, BertEncoder

class EncCNN1d(nn.Module):
    def __init__(self, input_dim=130, channel=128, dropout=0.1):
        super(EncCNN1d, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(input_dim, channel, 10, stride=2, padding=4),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel, channel*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(channel*2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel*2, channel*4, 5, stride=2, padding=2),
            nn.BatchNorm1d(channel*4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(channel*4, channel*2, 3, stride=1, padding=1),
        )
        self.dropout = nn.Dropout(dropout)
        self.phoneme_dim = 130

    def forward(self, input_batch):
        # input_batch of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(input_batch.transpose(1, 2))
        out = out.transpose(1, 2)  # to (batch x seq x dim)
        out = self.dropout(out)
        print(out.shape)
        return out

class SpeechEncoderBertModel(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config):
        super().__init__(config)
        self.speechfront = EncCNN1d()
        self.encoder = BertEncoder(config) # transformer based encoder
        # build audio position embeddings = 128
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        # due to the viseme is 512 and trans to hidden_size
        self.affine_layer = nn.Linear(self.speechfront.phoneme_dim, 
                                    config.hidden_size, bias=True)
        self.apply(self.init_weights)

    def forward(self, inputbatch, attention_mask, output_all_encoded_layers=False):
        print(f'[Debug] inputbatch {inputbatch.shape}')
        a_phonemes = self.speechfront(inputbatch)
        self.a_phonemes = self.affine_layer(a_phonemes)
        print(f'[Debug] affined v_visemes {self.v_visemes.shape}') # [Debug] v_visimes torch.Size([1, 4, 768])
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(f'[Debug] extended_attention_mask {extended_attention_mask}') # [Debug] [1, 1, 1, 4]

        # compute embedding 
        position_ids = torch.arange(0, inputbatch.size(1), dtype=torch.long).unsqueeze(0)
        # print("[Debug] position_ids {}".format(position_ids))
        position_embeddings = self.position_embeddings(position_ids)
        # print('position_embeddings {}'.format(position_embeddings.shape)) # torch.Size([1, 4, 768])
        embedding_output = self.v_visemes + position_embeddings
        # print('[Debug] embedding_output {}'.format(embedding_output.shape))  ## torch.Size([1, 4, 768])
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

if __name__ == '__main__':
    config_path = '/data7/MEmoBert/code/uniter3flow/config/uniter-speech_enc.json'
    config = BertConfig(config_path)
    model = SpeechEncoderBertModel(config)
    input = torch.Tensor(1, 10, 130) # (batchsize, seq_len, ft-dim)
    attention_mask = torch.tensor([1,1,1,1,1,1,1,0,0,0]).unsqueeze(0)
    encoded_layers = model.forward(input, attention_mask)
    print('encoded_layers {}'.format(encoded_layers.shape))