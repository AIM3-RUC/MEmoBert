import torch
from torch import nn 

class WavEncoder(nn.Module):
    def __init__(self):
        super(WavEncoder, self).__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        # wav_data: one channle and 16k sample rate 
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)

if __name__ == '__main__':
    net = WavEncoder()
    inp = torch.ones(32, 6 * 16000)
    out = net(inp)
    print(out.shape)
