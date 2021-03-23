'''
采用 Conv1D + Transformer 的结构作为语音的 Encoder.
输入是 130 维度的 ComparE 特征
'''

import torch
import os
from torch import nn
import torch.nn.functional as F
from code.uniter3flow.model.layer import TransformerEncoder


class EncCNN1d(nn.Module):
    '''
    use conv1d to
    '''
    def __init__(self, input_dim=130, channel=128, dropout=0.3):
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
        self.dp = nn.Dropout(dropout)

    def forward(self, wav_data):
        # wav_data of shape [bs, seq_len, input_dim]
        out = self.feat_extractor(wav_data.transpose(1, 2))
        out = out.transpose(1, 2)       # to (batch x seq x dim)
        out = self.dp(out)
        return out  


class CnnTransformerModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--a_num_hidden_layers', type=int, default=130)
        parser.add_argument('--a_num_attention_heads', type=int, default=128)
        parser.add_argument('--a_max_position_embeddings', type=int, default=4)
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE']
        self.model_names = ['enc', 'rnn', 'C']
        self.netenc = EncCNN1d(opt.input_dim, opt.enc_channel)
        self.netrnn = TransformerEncoder(opt.enc_channel*2, opt.num_layers, opt.nhead, opt.dim_feedforward)
        cls_layers = [int(x) for x in opt.cls_layers.split(',')] + [opt.output_dim]
        self.netC = FcEncoder(opt.enc_channel*2, cls_layers, dropout=0.3)
            
        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.998)) # 0.999
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.signal = input['A_feat'].to(self.device)
        self.label = input['label'].to(self.device)
        self.input = input

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.segments = self.netenc(self.signal)
        self.feat, _ = self.netrnn(self.segments)
        self.logits = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        loss = self.loss_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5.0) # 0.1

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()   
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step() 