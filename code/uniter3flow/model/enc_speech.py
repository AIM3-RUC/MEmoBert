"""
ComparE + Conv1D + Transformer
关于帧和音素的关系:
每帧的长度足够短来保证帧内信号是平稳的。
口型的变化是导致信号不平稳的原因，所以在一帧的期间内口型不能有明显变化，即一帧的长度应当小于一个音素的长度。
正常语速下，音素的持续时间大约是 50~200 毫秒，所以帧长一般取为小于 50 毫秒。
所以还是可以采用 Conv3D 的方式来获取 phoneme-level 的feature. 但是跟face一样的话就会导致序列过长 100frames/second.
Conv3D + Conv1D + Transformer
目前采用的 frame-level 0.060win, and shift=0.010, (200-50)/10=15 因此我们卷 15 frames, 作为一个 phoneme-level feature.

因为模型中有 Conv1D 所以没法使用 Attention Mask. 因此将AttentionMask全部置为1就行。
"""

from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.uniter3flow.model.model_base import BertConfig, BertPreTrainedModel, BertEncoder

class EncCNN1d(nn.Module):
    def __init__(self, input_dim=130, channel=256, dropout=0.1):
        super(EncCNN1d, self).__init__()
        '''
        input shape: (bs, input_dim, seq_len)
        return, con1d_output_dim=256, and the time-step will / 8
        '''
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
        self.con1d_output_dim = 512

    def forward(self, input_batch):
        # input_batch of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(input_batch.transpose(1, 2))
        out = out.transpose(1, 2)  # to (batch x seq x dim)
        out = self.dropout(out)
        return out

class EncCNN3DCNN1d(nn.Module):
    def __init__(self, input_dim=130, init_channel=256, dropout=0.1):
        super(EncCNN3DCNN1d, self).__init__()
        '''
        input shape: (bs, input_dim, seq_len)
        return con1d_output_dim=init_channel*2, and the time-step will / 8
        '''
        # conv3d-input, (batch-size, channle, timesteps, 130)
        self.frontend3D = nn.Sequential(
                            nn.Conv2d(1, init_channel, kernel_size=(15, 5), stride=(1, 2), padding=(2, 4), bias=False),
                            nn.BatchNorm2d(init_channel, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0,1))
                        )
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(input_dim, init_channel, 10, stride=2, padding=4),
            nn.BatchNorm1d(init_channel),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(init_channel, init_channel*2, 5, stride=2, padding=2),
            nn.BatchNorm1d(init_channel*2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(init_channel*2, init_channel*4, 5, stride=2, padding=2),
            nn.BatchNorm1d(init_channel*4),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(init_channel*4, init_channel*2, 3, stride=1, padding=1),
        )
        self.dropout = nn.Dropout(dropout)
        self.con1d_output_dim = 512

    def forward(self, input_batch):
        print('[Debug] input_batch {}'.format(input_batch.shape))
        # input of shape [bs, seq_len, input_dim]
        output = self.frontend3D(input_batch.unsqueeze(1)) # [bs, 1, seq_len, input_dim]
        print('[Debug] conv3d output {}'.format(output.shape))
        # input of shape [bs, seq_len, input_dim]
        output = output.squeeze(1).transpose(1, 2)
        print('[Debug] conv1d input {}'.format(output.shape))
        out = self.feat_extractor(output)
        print('[Debug] conv1d output {}'.format(out.shape))
        out = out.transpose(1, 2)  # to (batch x seq x dim)
        out = self.dropout(out)
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
        self.config = config
        self.speechfront = EncCNN1d()
        self.speech_encoder = BertEncoder(config) # transformer based encoder
        # build audio position embeddings = 128
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        # due to the viseme is 512 and trans to hidden_size
        self.affine_layer = nn.Linear(self.speechfront.con1d_output_dim, 
                                    config.hidden_size, bias=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.apply(self.init_weights)

    def forward(self, batch, output_all_encoded_layers=False):
        inputbatch = batch['speech_feat']
        position_ids = batch['speech_position_ids']
        attention_mask = batch['speech_attn_masks']
        # print(f'[Debug] inputbatch {inputbatch.shape}')
        output = self.speechfront(inputbatch)
        affine_a_output = self.affine_layer(output)
        # print(f'[Debug] affined a_output {self.a_output.shape}') # torch.Size([1, seq-len/8, 768])

        if self.config.add_cls_token:
            ## add the cls token on time dimension of output of the frontend.
            cls_token = repeat(self.cls_token, '() n d -> b n d', b = affine_a_output.size(0))
            # print(f'[Debug] cls_token {cls_token.shape}') # [Debug] cls_token torch.Size([1, 5, 768])
            affine_a_output = torch.cat((cls_token, affine_a_output), dim=1)

        # donnot sure the speech downsample, so set all to be one
        extended_attention_mask = torch.ones((affine_a_output.size(0), affine_a_output.size(1)))
        # print('[Debug] extended_attention_mask {}'.format(extended_attention_mask.shape)) # torch.Size([1, 2])

        position_embeddings = self.position_embeddings(position_ids)
        # print('position_embeddings {}'.format(position_embeddings.shape)) # torch.Size([1, 4, 768])
        embedding_output = affine_a_output + position_embeddings
        # print('[Debug] embedding_output {}'.format(embedding_output.shape))  ## torch.Size([1, 4, 768])
        encoded_layers = self.speech_encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, extended_attention_mask

if __name__ == '__main__':
    config_path = '/data7/MEmoBert/code/uniter3flow/config/uniter-speech_enc.json'
    config = BertConfig(config_path)
    model = SpeechEncoderBertModel(config)
    input = torch.Tensor(1, 300, 130) # (batchsize, seq_len, ft-dim)
    encoded_layers = model.forward(input)
    print('encoded_layers {}'.format(encoded_layers.shape))