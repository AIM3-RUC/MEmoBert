import os
import torch
import pandas as pd
import soundfile as sf
import numpy as np
from utils import get_basename, mkdir
import scipy.signal as spsig
from fairseq.models.wav2vec import Wav2VecModel

from .base_worker import BaseWorker

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
        #     self.print('Get audio from {}, save to {}'.format(video_path, save_path))
        # else:
        #     self.print('Found in {}, skip'.format(save_path))

        return save_path

        
class ComParEExtractor(BaseWorker):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, downsample=10, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        super().__init__()
        mkdir(tmp_dir)
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        utt_id = wav.split('/')[-2]
        save_path = os.path.join(self.tmp_dir, utt_id+'_'+get_basename(wav)+".csv")
        # if not os.path.exists(save_path):
        cmd = 'SMILExtract -C ~/opensmile-2.3.0/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        os.system(cmd.format(wav, save_path))
        
        df = pd.read_csv(save_path, delimiter=';')
        # timestamp = np.array(df['frameTime'])
        wav_data = df.iloc[:, 2:]
        if len(wav_data) > self.downsample:
            wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
            # self.print(f'Extract comparE from {wav}: {wav_data.shape}')
            if self.no_tmp:
                os.remove(save_path) 
            
        else:
            wav_data = None
            self.print(f'Error in {wav}, no feature extracted')
        return wav_data


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
            timestamp = int(timestamp / 1000 * sr)
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

class Wav2vecExtractor(BaseWorker):
    # 暂时先不用这个了
    ''' Wav2vec feature extractor
        downsample: downsample rate. Raw feature has 10ms step 
    '''
    def __init__(self, device=0, downsample=10, seg_len=0.25, step_size=0.1,
            pretrained_path="tools/wav2vec/pretrained_model/wav2vec_large.pt"):
        super().__init__()
        self.downsample = downsample
        self.pretrained_path = pretrained_path
        self.device = torch.device(f'cuda:{device}')
        self.model = self.get_pretrained_model()
    
    def get_pretrained_model():
        cp = torch.load(self.pretrained_path)
        model = Wav2VecModel.build_model(cp['args'], task=None)
        model.load_state_dict(cp['model'])
        model.eval()
        model.to(self.device)
        return model
    
    def __call__(self, wav_path):
        pass


if __name__ == '__main__':
    get_audio = AudioSplitor('./test_audio')
    extract_comparE = ComParEExtractor()
    vggish_extract = VggishExtractor()
    audio_path = get_audio("../resources/output1.mkv")
    comparE = extract_comparE(audio_path)
    vggish = vggish_extract(audio_path)
    print('comparE:', comparE.shape)
    print('vggish:', vggish.shape)
