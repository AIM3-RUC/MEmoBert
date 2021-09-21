import os
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
from preprocess.utils import get_basename, mkdir
import librosa
import scipy.signal as spsig
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from preprocess.tools.hook import MultiLayerFeatureExtractor
from preprocess.tasks.base_worker import BaseWorker

class AudioSplitor(BaseWorker):
    ''' 把语音从视频中抽出来存在文件夹中
        eg: 输入视频/root/hahah/0.mp4, save_root='./test/audio'
            输出音频位置: ./test/audio/hahah/0.wav (注意第24行, 按需求可修改)
            保存的采样率是16000, 16bit, 如需修改请参考30行: _cmd = "ffmpeg -i ...."
    '''
    def __init__(self, save_root, logger=None):
        super().__init__()
        self.audio_dir = save_root
        self.logger = logger

    def __call__(self, video_path):
        basename = get_basename(video_path)
        movie_name = video_path.split('/')[-2]
        save_dir = os.path.join(self.audio_dir, movie_name)
        mkdir(save_dir) 
        save_path = os.path.join(save_dir, basename + '.wav')
        if not os.path.exists(save_path):
            _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(video_path, save_path)
            os.system(_cmd)
        return save_path

class AudioSplitorTool(BaseWorker):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def __call__(self, video_path, save_path):
        if not os.path.exists(save_path):
            _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(video_path, save_path)
            os.system(_cmd)
        return save_path
    
class ComParEExtractor(BaseWorker):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=-1, tmp_dir='/data7/emobert/comparE_feature/raw_fts', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
            downsample. if =-1, then use the raw comparE fts, else use the resampeld fts.
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/data2/zjm/tools/opensmile-3.0-linux-x64'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, full_wav_path):
        # such as: /data7/emobert/data_nomask_new/audio_clips/No0079.The.Kings.Speech/188.wav
        movie_name = full_wav_path.split('/')[-2]
        basename = movie_name + '_' + os.path.basename(full_wav_path).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/compare16/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        # os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        p = subprocess.Popen([cmd.format(self.opensmile_tool_dir, full_wav_path, save_path)], stderr=subprocess.PIPE, shell=True)
        err = p.stderr.read()
        if err:
            raise RuntimeError(err)
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_ft_data = np.array(df.iloc[:, 2:])
        if self.downsample > 0:
            if len(wav_ft_data) > self.downsample:
                wav_ft_data = spsig.resample_poly(wav_ft_data, up=1, down=self.downsample, axis=0)
                if self.no_tmp:
                    os.remove(save_path) 
            else:
                raise ValueError('Error in {wav}, signal length must be longer than downsample parameter')
        return wav_ft_data
        

class VggishExtractor(BaseWorker):
    ''' 抽取vggish特征, 输入音频路径, 输出npy数组, 每帧128d
    '''
    def __init__(self, seg_len=0.1, step_size=0.1, device=0):
        ''' Vggish feature extractor
            seg_len: window size(with expansion of 1s, padding 0.5s at both sides)
            step_size: step size of each window
            device: GPU number
        '''
        super().__init__()
        self.seg_len = seg_len
        self.step_size = step_size
        self.device = torch.device(f'cuda:{device}')
        self.model = self.get_pretrained_vggish()
    
    def read_wav(self, wav_path):
        wav_data, sr = sf.read(wav_path, dtype='int16')
        return wav_data, sr
    
    def get_pretrained_vggish(self):
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        model.eval()
        model.postprocess = False
        model.device = self.device
        model.to(self.device)
        return model
    
    def get_vggish_segment(self, wav_data, sr, timestamps):
        block_len = int(0.98 * sr) ## vggish block is 0.96s, add some padding
        self.seg_len = int(self.seg_len * sr) ## 250ms
        pad_context = (block_len - self.seg_len) // 2
        ans = []
        for timestamp in timestamps:
            # timestamp = int(timestamp / 1000 * sr)
            timestamp = int(timestamp * sr) #hzp modified: 这里timestamp的单位应该是s而不是ms
            if timestamp >= len(wav_data) + pad_context: # 提供的部分音频长度比label的timestamp短
                cur_time_wav_data = np.array([wav_data[-1]] * block_len)
            else:                                        # 正常情况, timestamp的时间没超过audio_length
                left_padding = np.array(max(0, (pad_context - timestamp)) * [wav_data[0]])
                right_padding = np.array(max(0, (timestamp + self.seg_len + pad_context) - len(wav_data)) * [wav_data[-1]])
                target_data = wav_data[max(0, timestamp-pad_context): timestamp + self.seg_len + pad_context]
                cur_time_wav_data = np.concatenate([left_padding, target_data, right_padding])
                cur_time_wav_data = np.array(cur_time_wav_data)
            ans.append(cur_time_wav_data)
        
        return np.array(ans)

    def __call__(self, wav_path):
        wav_data, sr = self.read_wav(wav_path)
        time_length = len(wav_data) / sr
        timestamps = [self.step_size * n for n in range(int(time_length / self.step_size))]
        segments = self.get_vggish_segment(wav_data, sr, timestamps)
        vggish_feature = list(map(lambda x: self.model.forward(x, sr).cpu().detach().numpy(), segments))
        vggish_feature = np.array(vggish_feature).squeeze()
        # self.print(f'Extract vggish from {wav_path}: {vggish_feature.shape}')  
        if len(vggish_feature) < 2 or vggish_feature.shape[0] == 0:
            return None

        return vggish_feature

class Wav2VecExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧768d
    '''
    def __init__(self, downsample=4, gpu=0, use_asr_based_model=False):
        self.downsample = downsample
        self.device = torch.device('cuda:{}'.format(gpu))
        if use_asr_based_model:
            print('[INFO] use asr based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        else:
            print('[INFO] use vanilla based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        
    @staticmethod
    def read_audio(wav_path):
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        if sr * 10 < len(speech):
            print(f'{wav_path} long than 10 seconds and clip {speech.shape}')
            speech = speech[:int(sr * 10)]
        return speech, sr

    def __call__(self, wav):
        input_values, sr = Wav2VecExtractor.read_audio(wav)
        input_values = self.processor(input_values, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            ft = self.model(input_values).last_hidden_state

        if self.downsample > 0:
            ft = torch.cat([
                torch.mean(ft[:, i:i+self.downsample], dim=1) for i in range(0, ft.shape[1], self.downsample)
            ], dim=0)
        return ft.cpu().numpy()

class Wav2VecCNNExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧768d
    '''
    def __init__(self, gpu=0, use_asr_based_model=False):
        self.device = torch.device('cuda:{}'.format(gpu))
        
        if use_asr_based_model:
            print('[INFO] use asr based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        else:
            print('[INFO] use vanilla based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)

        self.mid_extractor = MultiLayerFeatureExtractor(self.model, ["feature_extractor"]) # feature_projection
        
    @staticmethod
    def read_audio(wav_path):
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        if sr * 10 < len(speech):
            print(f'{wav_path} long than 10 seconds and clip {speech.shape}')
            speech = speech[:int(sr * 10)]
        return speech, sr

    def __call__(self, wav):
        input_values, sr = Wav2VecExtractor.read_audio(wav)
        input_values = self.processor(input_values, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            _ = self.model(input_values).last_hidden_state
            ft = self.mid_extractor.extract()
        local_cnn_ft = ft[0][0].transpose(-1, -2).cpu().numpy()
        # print(local_cnn_ft.shape)
        return local_cnn_ft

class RawWavExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧768d
    '''
    def __init__(self, model_path, max_seconds=8):
        self.sr = 16000
        self.max_seconds = max_seconds
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
    def read_audio(self, wav_path):
        speech, sr = sf.read(wav_path)
        if sr != self.sr:
            speech = librosa.resample(speech, sr, self.sr)
            sr = self.sr
        if sr * self.max_seconds < len(speech):
            print(f'{wav_path} long than 10 seconds and clip {speech.shape}')
            speech = speech[:int(sr * self.max_seconds)]
        return speech, sr

    def __call__(self, wav_path):
        input_values, sr = self.read_audio(wav_path)
        input_values = self.processor(input_values, return_tensors="np", sampling_rate=sr).input_values
        return input_values[0]

if __name__ == '__main__':
    # get_audio = AudioSplitor('./test_audio')
    # extract_comparE = ComParEExtractor()
    # vggish_extract = VggishExtractor()
    # audio_path = get_audio("../resources/output1.mkv")
    # comparE = extract_comparE(audio_path)
    # vggish = vggish_extract(audio_path)
    # print('comparE:', comparE.shape)
    # print('vggish:', vggish.shape)

    audio_path = "/data7/emobert/data_nomask_new/audio_clips/No0001.The.Shawshank.Redemption/9.wav"
    # extract_wav2vec = Wav2VecExtractor(downsample=-1, gpu=7)
    # ft = extract_wav2vec(audio_path)
    # print(ft.shape)
    model_path = '/data7/MEmoBert/emobert/resources/pretrained/wav2vec_base'
    extract_wav = RawWavExtractor(model_path, max_seconds=8)
    input_values = extract_wav(audio_path)
    print(input_values.shape)