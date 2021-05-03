import logging
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

logger = logging.getLogger(__name__)

'''
wav2vec2.0 作为 Encoder. 
输入是经过 transformers.Wav2Vec2Processor 处理过的语音数据 (batch, dim)
这些加载 wav2vec2.0 模型的config文件
'''

class SpeechWav2Vec2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert isinstance(config, Wav2Vec2Config)   
        self.config = config            
        self.encoder = Wav2Vec2Model(config)

    def forward(self, batch):
        '''
        return output object:
        output.last_hidden_state: torch.FloatTensor = None
        output.hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        output.attentions: Optional[Tuple[torch.FloatTensor]] = None
        '''
        speech_feat = batch['speech_feat']
        output = self.encoder(speech_feat,
                                    attention_mask=None,
                                    output_attentions=None,
                                    output_hidden_states=None,
                                    return_dict=None)
        encoded_layers = output.last_hidden_state # torch.Size([1, 177, 768])
        return encoded_layers

if __name__ == '__main__':
    from transformers import Wav2Vec2Processor, Wav2Vec2Config
    import soundfile as sf
    import os, json
    model_path = '/data7/MEmoBert/emobert/resources/pretrained/wav2vec_base'
    model_config_path = '/data7/MEmoBert/emobert/resources/pretrained/wav2vec_base/config.json'
    pt_model_path = '/data7/MEmoBert/emobert/resources/pretrained/wav2vec_base/wav2vec_base.pt'
    audio_path = "/data7/emobert/data_nomask_new/audio_clips/No0001.The.Shawshank.Redemption/9.wav"
    device = torch.device('cuda:{}'.format(0))
    #### method of hugging face
    # config = None
    # processor = Wav2Vec2Processor.from_pretrained(model_path)
    # wav2vec_model = SpeechWav2Vec2Model(config, model_path)

    ## save as torch format, 保存的是一致的啊
    # torch.save(wav2vec_model.encoder.state_dict(), pt_model_path)
    # print('save model {}'.format(pt_model_path))
    # print(list(wav2vec_model.encoder.state_dict().keys()))
    # print(wav2vec_model.encoder.state_dict()['feature_extractor.conv_layers.0.conv.weight'])

    ### method of tranditional method
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    config = json.load(open(model_config_path, 'r'))
    config = Wav2Vec2Config().from_dict(config)
    wav2vec_model = SpeechWav2Vec2Model(config)
    wav2vec_model.encoder.load_state_dict(torch.load(pt_model_path))

    ### save as hugging face format
    # processor.save_pretrained(model_path)
    # wav2vec_model.encoder.save_pretrained(model_path)

    wav2vec_model.eval() # Note Important
    wav2vec_model.to(device)
    speech, sr = sf.read(audio_path)
    speech = speech[:50000] # 并不是严格的400/segment.
    print(f'speech {len(speech)} and sr {sr}')
    speech_processor = processor(speech, return_tensors="pt", sampling_rate=sr)
    print(f'speech_processor {speech_processor.input_values.shape}')
    speech_feat = speech_processor.input_values.to(device)
    inputbatch = {}
    inputbatch['speech_feat'] = speech_feat
    output = wav2vec_model(inputbatch)
    print(f'output {output.shape}')
    # 如何估算输出的长度？--- 
    # This results in an encoder output frequency of 49 hz with a stride of about 20ms between each sample, and a receptive field of 400 input samples or 25ms of audio. 
    out_len = len(speech) / (0.02 * 16000)
    print(f'out len {out_len}')