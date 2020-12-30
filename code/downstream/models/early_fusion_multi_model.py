
import torch
import torch.nn.functional as F
from torch import nn 
from torch.nn import CrossEntropyLoss
from .networks.lstm_encoder import LSTMEncoder
from .networks.textcnn_encoder import TextCNN
from .networks.classifier import FcClassifier
from .networks.tools import init_weights

class EarlyFusionMultiModel(nn.Module):
    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(EarlyFusionMultiModel, self).__init__()
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.device = torch.device("cuda:{}".format(opt.gpu_id))
        self.loss_names = ['CE']
        self.modality = opt.modality
        fusion_layers = list(map(lambda x: int(x), opt.mid_fusion_layers.split(',')))
        fusion_size = 0
        
        # acoustic model
        if 'A' in self.modality:
            self.netV = LSTMEncoder(opt.a_input_size, opt.a_hidden_size, opt.a_embd_method)
            fusion_size += opt.a_hidden_size

        # lexical model
        if 'L' in self.modality:
            self.netL = TextCNN(opt.l_input_size, opt.l_hidden_size)
            fusion_size += opt.l_hidden_size

        # visual model
        if 'V' in self.modality:
            self.netV = LSTMEncoder(opt.v_input_size, opt.v_hidden_size, opt.v_embd_method)
            fusion_size += opt.v_hidden_size
        
        self.netC = FcClassifier(fusion_size, fusion_layers, opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.criterion_ce = CrossEntropyLoss()
        
        # 全都采用默认的初始化方法
        # self.apply(init_weights)

    def set_input(self, batch):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if "A" in self.modality:
            self.acoustic = batch['acoustic'].float().to(self.device)
        if "L" in self.modality:
            self.lexical = batch['lexical'].float().to(self.device)
        if "V" in self.modality:
            self.visual = batch['visual'].float().to(self.device)
        self.label = batch['label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            final_embd.append(self.feat_A)
            
        if 'L' in self.modality:
            self.feat_L = self.netL(self.lexical)
            final_embd.append(self.feat_L)
        
        if 'V' in self.modality:
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
