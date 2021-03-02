import torch
from torch import tensor
import torch.nn.functional as F
from torch import nn 
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

'''
ResNet18,  https://github.com/afourast/deep_lip_reading
https://github.com/pytorch/vision/blob/master/torchvision/models/
参考torch官方的实现
'resnet18','resnet34','resnet50':
'''

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        '''
        inplanes: in channel number
        planesL out channel number 
        '''
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4
    def __init__(self, inplanes, planes, downsample, stride=1, groups=1, base_width= 64,  dilation=1):
        super(Bottleneck, self).__init__()
        '''
        downsample: None or nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        '''
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, gpu_id, block_type, resblocks, num_classes=8, zero_init_residual=False,
                    groups=1,  width_per_group=64, frozen_dense_blocks=0, **kwargs):
        super(ResNet, self).__init__()
        self.device = torch.device("cuda:{}".format(gpu_id))

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # first conv, gray image, only has one channel
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(self.inplanes)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(2, stride=2, padding=1)) 
        ]))
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        if block_type == 'basic':
            print('[Info] Use basic block')
            block = BasicBlock
        else:
            block = Bottleneck
        temp_inplanes = self.inplanes
        for i, num_layers in enumerate(resblocks):
            stride = 1 if i == 0 else 2
            if i <= frozen_dense_blocks - 1 and i >= 0:
                print("the {} block is fixed".format(i))
                with torch.no_grad():
                    self.features.add_module('resblock{}'.format(i+1), self._make_layer(block, temp_inplanes, num_layers, stride=stride))
            else:
                self.features.add_module('resblock{}'.format(i+1), self._make_layer(block, temp_inplanes, num_layers, stride=stride))
            temp_inplanes = temp_inplanes * 2
        # self.layer1 = self._make_layer(block, 64, resblocks[0])
        # self.layer2 = self._make_layer(block, 128, resblocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, resblocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, resblocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        self.criterion = CrossEntropyLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)
    
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
        x = self.features(self.bs_images)
        # print('[debug] after block4 {}'.format(x.shape))
        x = self.avgpool(x)
        # print('[debug] after block4 and avgpool {}'.format(x.shape))
        self.out_ft = torch.flatten(x, 1)  # ([64, 512])
        # print('[debug] final fc {}'.format(self.out_ft.shape))
        logits = self.classifier(self.out_ft)
        self.pred = F.softmax(logits, dim=-1)
        self.loss = self.criterion(logits, self.bs_labels)
        
    def backward(self, max_grad=0.0):
        """Calculate the loss for back propagation"""
        self.loss.backward()
        if max_grad > 0:
            print('[Debug] Use clip_grad_norm')
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad)

class ResNetEncoder(nn.Module):
    def __init__(self, block_type, resblocks, num_classes=8, zero_init_residual=False,
                    groups=1,  width_per_group=64, frozen_dense_blocks=0, **kwargs):
        super(ResNetEncoder, self).__init__()
        self.frozen_dense_blocks = frozen_dense_blocks
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # first conv, gray image, only has one channel
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(self.inplanes)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=2, stride=2, padding=1)) 
        ]))
        ## same as above
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        if block_type == 'basic':
            print('[Info] Use basic block')
            block = BasicBlock
        else:
            block = Bottleneck
        # feature maps for 4 blocks: 64, 128, 256, 512
        temp_inplanes = self.inplanes
        for i, num_layers in enumerate(resblocks):
            stride = 1 if i == 0 else 2
            if i <= frozen_dense_blocks - 1 and i >= 0:
                print("the {} block is fixed".format(i))
                with torch.no_grad():
                    self.features.add_module('resblock{}'.format(i+1), self._make_layer(block, temp_inplanes, num_layers, stride=stride))
            else:
                self.features.add_module('resblock{}'.format(i+1), self._make_layer(block, temp_inplanes, num_layers, stride=stride))
            temp_inplanes = temp_inplanes * 2
        ## same as above 
        # self.layer1 = self._make_layer(block, 64, resblocks[0])
        # self.layer2 = self._make_layer(block, 128, resblocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, resblocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, resblocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, images):
        images = images.unsqueeze(1) # (T, 1, H, W)
        x = self.features(images)
        x = self.avgpool(x)
        out_ft = torch.flatten(x, 1)  # ([64, 512])
        self.v_emb = out_ft
        return out_ft