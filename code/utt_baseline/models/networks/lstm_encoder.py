import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    ''' LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_size):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embd_size = embd_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.embd_size),
            nn.ReLU(),
        )

    def forward(self, x, length):
        batch_size = x.size(0)
        # x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        r_out, (h_n, h_c) = self.rnn(x)
        h_n = h_n.contiguous().view(batch_size, -1)
        embd = self.fc(h_n)
        return embd

class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last', pool_len=128):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method
        self.pool_len = pool_len

        if self.embd_method == 'maxpool':
            self.maxpool = nn.MaxPool1d(self.pool_len)
        
        elif self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文：Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # in_data = r_out.transpose(1,2)
        # embd = F.max_pool1d(in_data, in_data.size(2))          # r_out.size()=>[batch_size, seq_len, hidden_size]
        # print(r_out.transpose(1,2).shape)                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        embd = self.maxpool(r_out.transpose(1,2))
        return embd.squeeze()

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze()

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)
        return embd

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, layer=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_size, num_layers=layer, bidirectional=True)
        self.decoder = nn.LSTM(2 * hidden_size, output_dim, num_layers=layer, bidirectional=True)
        self.fc_enc = nn.Linear(2* hidden_size, hidden_size)
        self.relu_enc = nn.ReLU()
        self.fc_dec = nn.Linear(hidden_size, 2 * hidden_size)
        self.relu_dec = nn.ReLU()
        self.fc_recon = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    
    def forward(self, x):
        latent, _ = self.encoder(x)
        latent = self.relu_enc(self.fc_enc(latent))
        dec_latent = self.relu_dec(self.fc_dec(latent))
        recon, _ = self.decoder(dec_latent)
        recon = self.fc_recon(recon)
        return recon, latent

if __name__ == '__main__':
    # a = LSTMEncoder(342, 128, 'attention')
    # print(a)
    # data = torch.Tensor(12, 20, 342)
    # print(a(data).shape)
    a = LSTMAutoencoder(300, 128, 130)
    print(a)
    data = torch.Tensor(20, 22, 300)
    recon, latent = a(data)
    print(recon.shape)
    print(latent.shape)

