import os, glob
import subprocess
import datetime
import h5py
import numpy as np
from tqdm import tqdm
from preprocess.utils import get_basename, mkdir
from preprocess.tasks.base_worker import BaseWorker

class VideoCutter(BaseWorker):
    ''' --discard on 2021/03/01 
    按句子的timestamp切分视频
        save_root: 切出来的视频放在哪里, 
        padding: 每句话两端的padding时间
        return: sub_video_dir
    '''
    def __init__(self, save_root, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.save_root = save_root

    def calc_time(self, time_str):
        assert len(time_str.split(':')) == 3, 'Time format should be %H:%M:%S, but got {}'.format(time_str)
        hour, minite, second = time_str.split(':')
        return float(hour) * 3600 + float(minite) * 60 + float(second)
    
    def strptime(self, seconds):
        hour, minite, second = 0, 0, 0
        hour = int(seconds // 3600)
        minite = int(seconds % 3600 // 60)
        second = seconds - hour*3600 - minite*60
        return f"{hour}:{minite}:{second}"
    
    def __call__(self, video_path, transcripts):
        basename = get_basename(video_path)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        # _cmd = 'ffmpeg -i {} -g 1 -keyint_min 1 -ss {} -t {} -c copy -copyts {} -y > /dev/null 2>&1' # -vcodec copy -acodec copy
        _cmd = 'ffmpeg -ss {} -t {} -i {} -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y >/dev/null 2>&1 '
        for i, info in enumerate(transcripts):
            save_path = os.path.join(save_dir, f"{i}.mp4")
            start = info["start"]
            end = info["end"]
            duration = self.calc_time(end) - self.calc_time(start)
            duration = self.strptime(duration)
            # print(_cmd.format(video_path, start, duration, save_path))
            os.system(_cmd.format(start, duration, video_path, save_path))
            print(_cmd.format(start, duration, video_path, save_path))
            input()
            # self.print(f"[{i}/{len(transcripts)}]From {video_path}, cut video of {start}->{end}")
        return save_dir

class VideoCutterOneClip(BaseWorker):
    ''' 按句子的timestamp切分视频, 一次切一个视频片段出来
        save_root: 切出来的视频放在哪里, 
        padding: 每句话两端的padding时间
        return: sub_video_dir
    '''
    def __init__(self, save_root, padding=0, **kwargs):
        super().__init__(**kwargs)
        self.save_root = save_root

    def calc_time(self, time_str):
        assert len(time_str.split(':')) == 3, 'Time format should be %H:%M:%S'
        hour, minite, second = time_str.split(':')
        return float(hour) * 3600 + float(minite) * 60 + float(second)
    
    def strptime(self, seconds):
        hour, minite, second = 0, 0, 0
        hour = int(seconds // 3600)
        minite = int(seconds % 3600 // 60)
        second = seconds - hour*3600 - minite*60
        return f"{hour}:{minite}:{second:.2f}"
    
    def __call__(self, video_path, transcript_info):
        basename = get_basename(video_path)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        # _cmd = 'ffmpeg -ss {} -i {} -to {} -c copy -copyts {} -y > /dev/null 2>&1'  # error one
        _cmd = 'ffmpeg -ss {} -t {} -i {} -c:v libx264 -c:a aac -strict experimental -b:a 180k {} -y >/dev/null 2>&1 '
        save_path = os.path.join(save_dir, f"{transcript_info['index']}.mp4")
        if not os.path.exists(save_path):
            start = transcript_info["start"]
            end = transcript_info["end"]
            duration = self.calc_time(end) - self.calc_time(start)
            duration = self.strptime(duration)
            os.system(_cmd.format(start, duration, video_path, save_path))
        return save_dir

class SaverPerVideo(BaseWorker):
    '''一个video保存一个文件'''
    def __init__(self, save_root, format='npy'):
        self.save_root = save_root
        assert format in ['npy', 'h5'], 'Supported save format:[npy, h5], {} not supported yet'
        self.format = format
    
    def __call__(self, video_path, audio_features, vision_features, text_features):
        ''' Save features to npy file
            audio_features: audio feature tensor [sub_video_num, time_len, D_a]
            vision_features: vision features [sub_video_num, time_len, D_v]
            text_features: text features    [sub_video_num, time_len, D_t]
            name: save to save_root/name.npy
        '''
        assert audio_features.shape[0] == vision_features.shape[0] == text_features.shape[0]
        basename = get_basename(video_path)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        if self.format == 'npy':
            np.save(os.path.join(save_dir, 'audio.npy'), audio_features)
            np.save(os.path.join(save_dir, 'vision.npy'), vision_features)
            np.save(os.path.join(save_dir, 'text.npy'), text_features)
        elif self.format == 'h5py':
            pass

class CompileFeatures(BaseWorker):
    ''' 将一个[(time_len, D), (time_len, D), ...] 的特征在时间维度上拼接并打包成一组npy向量，同时返回每个单位的长度
    '''
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def __call__(self, feature_list):
        lengths = np.array([len(x) for x in feature_list])
        ans = np.concatenate(feature_list, axis=0)
        return ans, lengths

class FilterClips(BaseWorker):
    ''' Filter out the video clips whose duration is shotter than threshold (second).
        video_clip: path to the target video clip
    '''
    def __init__(self, threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def calc_time(self, time_str):
        assert len(time_str.split(':')) == 3, 'Time format should be %H:%M:%S, but got {}'.format(time_str)
        hour, minite, second = time_str.split(':')
        return float(hour) * 3600 + float(minite) * 60 + float(second)
    
    def __call__(self, video_clip):
        p = subprocess.Popen([f'ffmpeg -i {video_clip}'], stderr=subprocess.PIPE, shell=True)
        lines = p.stderr.readlines()
        duration_time = -1
        for line in lines:
            line = line.decode().strip()
            if line.startswith('Duration:'):
                duration = ':'.join(line.split(',')[0].split(':')[1:]).strip()
                duration_time = self.calc_time(duration)
        if duration_time == -1:
            raise RuntimeError('[FilterClips]: Can not found duration information in {}'.format(video_clip))
        else:
            return True if duration_time >= self.threshold else False

def pool_filter(cond, iterable, pool):
    # list(tqdm(pool.imap(get_frames, all_video_clip), total=len(all_video_clip)))
    iterable = list(iterable)
    flags = list(tqdm(pool.imap(cond, iterable), total=len(iterable)))
    ret = filter(lambda x: x[1], zip(iterable, flags))
    ret = list(map(lambda x: x[0], ret))
    return ret

class FilterTranscrips(BaseWorker):
    ''' Filter out the video clips whose duration is shotter than threshold (second).
        video_clip: path to the target video clip
    '''
    def __init__(self, threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
    
    def calc_time(self, time_str):
        assert len(time_str.split(':')) == 3, 'Time format should be %H:%M:%S, but got {}'.format(time_str)
        hour, minite, second = time_str.split(':')
        return float(hour) * 3600 + float(minite) * 60 + float(second)
    
    def __call__(self, transcript):
        end = self.calc_time(transcript['end'])
        start = self.calc_time(transcript['start'])
        duration_time = end - start
        return True if duration_time >= self.threshold else False

if __name__ == '__main__':
    from tasks.text import TranscriptPackager
    package_transcript = TranscriptPackager()
    ass_path = 'data/transcripts/No0001.The.Shawshank.Redemption.ass'
    transcripts = package_transcript(ass_path)
    cut_video = VideoCutter('test_video_cut')
    cut_video("../resources/raw_movies/No0001.The.Shawshank.Redemption.mkv", transcripts[:10])
    # filter_fun = FilterClips(10)
    # print(filter_fun("/data6/zjm/emobert/resources/output1.mkv"))

