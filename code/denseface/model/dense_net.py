import torch
from torch import tensor
import torch.nn.functional as F
from torch import nn 
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

'''
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
参考torch官方的实现
'''

class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) 
    growth_rate: 每层有多少个 kernel 的个数
    bn_size: 
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    """DenseBlock
    一个denseblock, block 内是 dense-connection.
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class DenseNet(nn.Module):
    "DenseNet-BC model"
    def __init__(self, gpu_id, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, num_classes=8, **kwargs):
        """
        default is the densenet121 setting
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d -- growth_rate*2
        :param bn_size: (int) the factor using in the bottleneck layer  --4
        :param compression_rate: (float) the compression rate used in Transition Layer --0.5
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNet, self).__init__()
        self.device = torch.device("cuda:{}".format(gpu_id))

        # first Conv2d # kernel_size=3, strides=[1, 2, 2, 1]
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False))
            # ("norm0", nn.BatchNorm2d(num_init_features)), # remove this 
            # ("relu0", nn.ReLU(inplace=True)), # remove this 
            # ("pool0", nn.MaxPool2d(3, stride=2, padding=1)) # remove this 
        ]))

        self.end_points = {}
        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = _Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm{}".format(len(block_config)), nn.BatchNorm2d(num_features))
        self.features.add_module("relu{}".format(len(block_config)), nn.ReLU(inplace=True))

        # classification layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.criterion = CrossEntropyLoss()

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
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
        features = self.features(self.bs_images) # torch.Size([64, 342, 8, 8])
        # print('dense output {}'.format(features.size())) 
        self.out_ft = F.avg_pool2d(features, kernel_size=8, stride=1).view(features.size(0), -1)  # torch.Size([64, 342])
        # print('out_ft {}'.format(out_ft.size()))
        logits = self.classifier(self.out_ft)
        # print('out logits {}'.format(logits.size()))    
        self.pred = F.softmax(logits, dim=-1)
        if getattr(self, 'bs_label', None):
            self.loss = self.criterion(logits, self.bs_labels)
        
    def backward(self, max_grad=0.0):
        """Calculate the loss for back propagation"""
        self.loss.backward()
        if max_grad > 0:
            print('[Debug] Use clip_grad_norm')
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad)

class DenseNetEncoder(nn.Module):
    "DenseNet-BC model"
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0, frozen_dense_blocks=0, **kwargs):
        """
        default is the densenet121 setting
        :param growth_rate: (int) number of filters used in DenseLayer, `k` in the paper
        :param block_config: (list of 4 ints) number of layers in each DenseBlock
        :param num_init_features: (int) number of filters in the first Conv2d -- growth_rate*2
        :param bn_size: (int) the factor using in the bottleneck layer  --4
        :param compression_rate: (float) the compression rate used in Transition Layer --0.5
        :param drop_rate: (float) the drop rate after each DenseLayer
        :param num_classes: (int) number of classes for classification
        """
        super(DenseNetEncoder, self).__init__()
        # first Conv2d # kernel_size=3, strides=[1, 2, 2, 1]
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=3, stride=2, padding=1, bias=False))
            # ("norm0", nn.BatchNorm2d(num_init_features)), # remove this 
            # ("relu0", nn.ReLU(inplace=True)), # remove this 
            # ("pool0", nn.MaxPool2d(3, stride=2, padding=1)) # remove this 
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Jinming add: only finetune last blocks
            if i <= frozen_dense_blocks - 1:
                print("the {} block is fixed".format(i))
                with torch.no_grad():
                    block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
                    self.features.add_module("denseblock%d" % (i + 1), block)
                    num_features += num_layers*growth_rate
                    if i != len(block_config) - 1:
                        transition = _Transition(num_features, int(num_features*compression_rate))
                        self.features.add_module("transition%d" % (i + 1), transition)
                        num_features = int(num_features * compression_rate)
            else:
                block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
                self.features.add_module("denseblock%d" % (i + 1), block)
                num_features += num_layers*growth_rate
                if i != len(block_config) - 1:
                    transition = _Transition(num_features, int(num_features*compression_rate))
                    self.features.add_module("transition%d" % (i + 1), transition)
                    num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm{}".format(len(block_config)), nn.BatchNorm2d(num_features))
        self.features.add_module("relu{}".format(len(block_config)), nn.ReLU(inplace=True))

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        # (T, H, W)
        images = images.unsqueeze(1) # (T, 1, H, W)
        features = self.features(images) # torch.Size([64, 342, 8, 8])
        # print('dense output {}'.format(features.size())) 
        out_ft = F.avg_pool2d(features, kernel_size=8, stride=1).view(features.size(0), -1)  # torch.Size([64, 342])
        return out_ft