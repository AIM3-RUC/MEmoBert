import logging
import torch
from torch import nn
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)

'''
wav2vec2.0 作为 Encoder. 
输入是经过 transformers.Wav2Vec2Processor 处理过的语音数据 (batch, dim)
    不需要 position-embeeding
'''

class SpeechEncoderBertModel(nn.Module):
    def __init__(self, config, fix_speech_branch=True):
        super().__init__()
        self.config = config    
        self.fix_speech_branch = fix_speech_branch
        if fix_speech_branch:
            logger.info('[INFO] Fix the speech branch!!!')
        else:
            logger.info('[INFO] Update the speech branch when training!!!')
        self.speech_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, batch):
        '''
        return output object:
        output.last_hidden_state: torch.FloatTensor = None
        output.hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        output.attentions: Optional[Tuple[torch.FloatTensor]] = None
        '''
        inputbatch = batch['speech_feat']
        if self.fix_speech_branch:
            with torch.no_grad():
                output = self.speech_encoder(inputbatch,
                                                attention_mask=None,
                                                output_attentions=None,
                                                output_hidden_states=None,
                                                return_dict=None)
        else:
            output = self.speech_encoder(inputbatch,
                                                attention_mask=None,
                                                output_attentions=None,
                                                output_hidden_states=None,
                                                return_dict=None)
        encoded_layers = output.last_hidden_state # torch.Size([1, 177, 768])
        return encoded_layers[0]


if __name__ == '__main__':
    import Wav2Vec2Processor
    import soundfile as sf
    audio_path = "/data7/emobert/data_nomask_new/audio_clips/No0001.The.Shawshank.Redemption/9.wav"
    device = torch.device('cuda:{}'.format(0))
    config = None
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model = SpeechEncoderBertModel(config)
    wav2vec_model.to(device)
    speech, sr = sf.read(audio_path)
    print(f'speech {len(speech)} and sr {sr}')
    speech_feat = processor(speech, return_tensors="pt", sampling_rate=sr).input_values.to(device)
    inputbatch = {}
    inputbatch['speech_feat'] = speech_feat
    wav2vec_model(inputbatch)