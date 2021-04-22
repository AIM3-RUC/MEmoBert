"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
Conv3D + Resnet  from:  https://github.com/lordmartian/deep_avsr
Transformers from: Add by Jinming
visual branch is Conv3D + Resnet + Transformers
"""
from einops import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.uniter3flowFOM.model.model_base import BertConfig, BertPreTrainedModel, BertEncoder

class ResNetLayer(nn.Module):
    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """
    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        if self.stride == 1:
            residualBatch = inputBatch
        else:
            residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch

class ResNet(nn.Module):
    """
    An 18-layer ResNet architecture.
    """
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 128, stride=2)
        self.layer3 = ResNetLayer(128, 256, stride=2)
        self.layer4 = ResNetLayer(256, 512, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        return

    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch

class ResNet3D(nn.Module):
    """
    for video emotion recognition. 保证原来的网络结构，输入大小为112*112.
    A visual feature extraction module. Generates a 512-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    # conv3d input: (batch-size, channle, timesteps, 112, 112)
    """
    def __init__(self):
        super(ResNet3D, self).__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet()
        self.viseme_dim = 512

    def forward(self, inputBatch):
        # inputBatch shape: (batchsize, timesteps, channle, H, W)
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        # print('[Debug] inputBatch {}'.format(inputBatch.shape))
        batchsize = inputBatch.shape[0] #Note: not-real-batchsize
        batch = self.frontend3D(inputBatch)
        # print('[Debug] Conv3d Output {}'.format(batch.shape))

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        # print('[Debug] Res18 Input {}'.format(batch.shape))
        outputBatch = self.resnet(batch)
        # print('[Debug] Res18 Output {}'.format(batch.shape))
        outputBatch = outputBatch.reshape(batchsize, -1, 512)
        # print('[Debug] Output feature {}'.format(outputBatch.shape))
        outputBatch = outputBatch.transpose(1 ,2)
        outputBatch = outputBatch.transpose(1, 2).transpose(0, 1)
        return outputBatch

class VisualEncoderBertModel(BertPreTrainedModel):
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
        self.visualfront = ResNet3D()
        self.encoder = BertEncoder(config) # transformer based encoder
        # build audio position embeddings = 128
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        # due to the viseme is 512 and trans to hidden_size
        self.affine_layer = nn.Linear(self.visualfront.viseme_dim, 
                                    config.hidden_size, bias=True)
        # add one cls token, 在CNN之后加, 即输入transformer的时候加
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.apply(self.init_weights)

    def forward(self, batch, output_all_encoded_layers=False):
        inputbatch = batch['img_feat']
        position_ids = batch['img_position_ids']
        attention_mask = batch['img_attn_masks']
        # print(f'[Debug] inputbatch {inputbatch.shape}')
        inputbatch = inputbatch.unsqueeze(2) # add channel
        v_visemes = self.visualfront(inputbatch) # torch.Size([1, 4, 512])
        affine_v_visemes = self.affine_layer(v_visemes)
        # print(f'[Debug] affined v_visemes {affine_v_visemes.shape}') # [Debug] v_visimes torch.Size([1, 4, 768])
        
        # compute self-attention mask.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
                                    dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(f'[Debug] extended_attention_mask {extended_attention_mask}') # [Debug] [1, 1, 1, 4]

        if self.config.add_cls_token:
            ## add the cls token on time dimension of output of the frontend.
            cls_token = repeat(self.cls_token, '() n d -> b n d', b = affine_v_visemes.size(0))
            # print(f'[Debug] cls_token {cls_token.shape}') # [Debug] cls_token torch.Size([1, 5, 768])
            affine_v_visemes = torch.cat((cls_token, affine_v_visemes), dim=1)
        # print("[Debug] position_ids {}".format(position_ids.shape))
        position_embeddings = self.position_embeddings(position_ids)
        # print('position_embeddings {}'.format(position_embeddings.shape)) # torch.Size([1, 5, 768]), add cls-token

        embedding_output = affine_v_visemes + position_embeddings
        # print('[Debug] embedding_output {}'.format(embedding_output.shape))  ## torch.Size([1, 5, 768]), add cls-token
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, extended_attention_mask

if __name__ == '__main__':
    # model = ResNet3D()
    # input = torch.Tensor(32, 10, 1, 112, 112)
    # model.forward(input)
    config_path = '/data7/MEmoBert/code/uniter3flow/config/uniter-visual_enc.json'
    config = BertConfig(config_path)
    model = VisualEncoderBertModel(config)
    input = torch.Tensor(1, 4, 1, 112, 112) # (batchsize, seq_len, channel, img-dim, imd-dim)
    attention_mask = torch.tensor([1,1,0,0]).unsqueeze(0)
    encoded_layers = model.forward(input, attention_mask)
    print('encoded_layers {}'.format(encoded_layers.shape))