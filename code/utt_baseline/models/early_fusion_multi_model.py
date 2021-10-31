
import torch
import torch.nn.functional as F
from torch import nn 
from torch.nn import CrossEntropyLoss
from .networks.lstm_encoder import LSTMEncoder
from .networks.fc_encoder import FcEncoder
from .networks.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_weights
from .networks.resnet3d import ResNet3D

class EarlyFusionMultiModel(nn.Module):
    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(EarlyFusionMultiModel, self).__init__()
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.opt = opt
        self.device = torch.device("cuda:{}".format(opt.gpu_id))
        self.loss_names = ['CE']
        self.modality = opt.modality
        fusion_layers = list(map(lambda x: int(x), opt.mid_fusion_layers.split(',')))
        fusion_size = 0
        
        # acoustic model
        if 'A' in self.modality:
            if 'IS10_norm' == opt.a_ft_type or opt.a_ft_type.startswith('sent'):
                print('Use FC encoder process the IS10/Sentence-Level features')
                self.netA = FcEncoder(opt.a_input_size, [opt.a_hidden_size]*2)
            else:
                self.netA = LSTMEncoder(opt.a_input_size, opt.a_hidden_size, opt.a_embd_method, pool_len=opt.max_acoustic_tokens)
            fusion_size += opt.a_hidden_size

        # text model
        if 'L' in self.modality:
            if opt.l_ft_type.startswith('sent'):
                print('Use FC encoder process the sentence-level text features')
                self.netL = FcEncoder(opt.l_input_size, [opt.l_hidden_size]*2)
            else:
                self.netL = TextCNN(opt.l_input_size, opt.l_hidden_size)
            fusion_size += opt.l_hidden_size

        # visual model
        if 'V3d' in self.modality:
            self.front3d = ResNet3D()
            self.netV = LSTMEncoder(opt.v_input_size, opt.v_hidden_size, opt.v3d_embd_method)
            fusion_size += opt.v_hidden_size
        elif 'V' in self.modality:
            if opt.v_ft_type.startswith('sent'):
                print('Use FC encoder process the sentence-level visual features')
                self.netV = FcEncoder(opt.v_input_size, [opt.v_hidden_size]*2)
            else:
                self.netV = LSTMEncoder(opt.v_input_size, opt.v_hidden_size, opt.v_embd_method, pool_len=opt.max_visual_tokens)
            fusion_size += opt.v_hidden_size
        
        self.netC = FcClassifier(fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.criterion_ce = CrossEntropyLoss()
        
        # 全都采用默认的初始化方法
        if self.opt.init_type != 'none':
            init_weights(self, init_type=self.opt.init_type)

    def set_input(self, batch):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if "A" in self.modality:
            self.acoustic = batch['acoustic'].float().to(self.device)

        if "L" in self.modality:
            self.text = batch['text'].float().to(self.device)

        if "V3d" in self.modality:
            # default (batchsize, timesteps=50, img-size, img-size) to (batchsize, timesteps=50, Channel=1, img-size, img-size)
            self.visual = batch['visual3d'].float().to(self.device)
        elif "V" in self.modality:
            self.visual = batch['visual'].float().to(self.device)

        self.label = batch['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            final_embd.append(self.feat_A)
            
        if 'L' in self.modality:
            self.feat_L = self.netL(self.text)
            final_embd.append(self.feat_L)
        
        if 'V3d' in self.modality:
            output = self.front3d(self.visual)
            self.feat_V = self.netV(output)
            final_embd.append(self.feat_V)
        elif 'V' in self.modality:
            self.feat_V = self.netV(self.visual)
            final_embd.append(self.feat_V)
        
        # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)
        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        self.loss = self.criterion_ce(self.logits, self.label)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.opt.grad_norm)