
import torch
from torch import nn

class BertPostagHead(nn.Module):
    '''
    # 同样是时序的，对于每个词的词性进行分类
    '''
    def __init__(self, config, pos_tag_embedding):
        super(BertPostagHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        
        self.decoder = nn.Linear(pos_tag_embedding.size(1),
                                 pos_tag_embedding.size(0),
                                 bias=False)
        self.decoder.weight = pos_tag_embedding
        self.bias = nn.Parameter(torch.zeros(pos_tag_embedding.size(0)))
    
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    
class BertSentiHead(nn.Module):
    def __init__(self, config, senti_word_embedding):
        super(BertSentiHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(senti_word_embedding.size(1),
                                 senti_word_embedding.size(0),
                                 bias=False)
        
        self.decoder.weight = senti_word_embedding
        self.bias = nn.Parameter(torch.zeros(senti_word_embedding.size(0)))
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
    
class BertPolarityHead(nn.Module):
    def __init__(self, config, polarity_embedding):
        super(BertPolarityHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(polarity_embedding.size(1),
                                 polarity_embedding.size(0),
                                 bias=False)
        
        self.decoder.weight = polarity_embedding
        self.bias = nn.Parameter(torch.zeros(polarity_embedding.size(0)))
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


