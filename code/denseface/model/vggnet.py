import torch
from torch import tensor
import torch.nn.functional as F
from torch import nn 
from torch.nn import CrossEntropyLoss
from collections import OrderedDict


class _ConvBlock(nn.Sequential):
    """DenseBlock
    一个denseblock, block 内是 dense-connection.
    """
    def __init__(self, num_input_features, num_out_features, drop_rate, pool_type=None, num_conv_layers=2, kernel_size=3):
        super(_ConvBlock, self).__init__()
        for i in range(num_conv_layers):
            self.add_module("conv{}".format(i+1), nn.Conv2d(num_input_features, num_out_features,
                                            kernel_size=kernel_size, stride=1, padding=1, bias=False))
            self.add_module("relu{}".format(i+1), nn.ReLU(inplace=True))
        if pool_type is None:
            pass
        elif pool_type == 'max':
            self.add_module("maxpool", nn.MaxPool2d(2, stride=2))
        elif pool_type == 'avg':
            self.add_module("avgpool", nn.AvgPool2d(2, stride=2))
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = super(_ConvBlock, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class VggNet(nn.Module):
    def __init__(self, gpu_id, drop_rate=0.25, num_classes=8, **kwargs):
        """
        default is the densenet121 setting
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(VggNet, self).__init__()
        self.device = torch.device("cuda:{}".format(gpu_id))

        self.conv_block1 =  _ConvBlock(1, 64, drop_rate, pool_type='max', num_conv_layers=2, kernel_size=3)
        self.conv_block2 =  _ConvBlock(64, 128, drop_rate, pool_type='max', num_conv_layers=2, kernel_size=3)
        self.conv_block3 =  _ConvBlock(128, 256, drop_rate, pool_type='max', num_conv_layers=3, kernel_size=3)
        self.conv_block4 =  _ConvBlock(256, 256, drop_rate, pool_type='max', num_conv_layers=3, kernel_size=3)
        self.conv_block5 =  _ConvBlock(256, 512, drop_rate, pool_type=None, num_conv_layers=1, kernel_size=4)
        self.conv_block6 =  _ConvBlock(512, 512, drop_rate, pool_type=None, num_conv_layers=1, kernel_size=1)
        self.conv_block7 =  _ConvBlock(512, 512, drop_rate, pool_type=None, num_conv_layers=1, kernel_size=1)
        
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        self.criterion = CrossEntropyLoss()

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def set_input(self, batch):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # (batch, heigh, width, channel)
        bs_images = batch['images'].float().to(self.device)
        # (batch, channel, heigh, width)
        self.bs_images = bs_images.permute(0, 3, 1, 2)
        if batch.get('labels') is not None:
            self.bs_labels = batch['labels'].to(self.device)

    def forward(self):
        #input-shape: (N, Cin, H, W) 
        output = self.conv_block1(self.bs_images)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        output = self.conv_block5(output)
        output = self.conv_block6(output)
        self.out_ft = self.conv_block7(output)

        logits = self.classifier(self.out_ft)
        # print('out logits {}'.format(logits.size()))    
        self.pred = F.softmax(logits, dim=-1)
        self.loss = self.criterion(logits, self.bs_labels)
        
    def backward(self, max_grad=0.0):
        """Calculate the loss for back propagation"""
        self.loss.backward()
        if max_grad > 0:
            print('[Debug] Use clip_grad_norm')
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad)
