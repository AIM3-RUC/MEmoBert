import os, glob
import shutil
import h5py
import json
import numpy as np
import multiprocessing
from tqdm import tqdm
from .utils import get_basename, mkdir
from .tasks.audio import *
from .tasks.vision import *
from .tasks.text import *
from. tasks.common import *

def save_h5(feature, lengths, save_path):
    h5f = h5py.File(save_path, 'w')
    h5f['feature'] = feature
    h5f['lengths'] = lengths
    h5f.close()

def vision_process(paths):
    frame_dir = './data/frames'
    face_dir = './data/faces'
    get_frames = Video2Frame(save_root=frame_dir)
    get_faces = VideoFaceTracker(save_root=face_dir)
    get_activate_spk = ActiveSpeakerSelector()
    select_faces = FaceSelector()
    fail_flag = False
    video_clip, audio_path = paths
    _frame_dir = get_frames(video_clip)
    _face_dir = get_faces(_frame_dir)
    active_spk_id = get_activate_spk(_face_dir, audio_path)
    if active_spk_id is None:
        fail_flag = True
        count['no_active_spk'] += get_basename(_frame_dir)
    else:
        face_paths = select_faces(face_dir, active_spk_id)
        if len(face_paths) < 0.4 * len(glob.glob(os.path.join(frame_dir, '*.jpg'))):
            fail_flag = True
            count['face_too_less'] += get_basename(_face_dir)
        
    # if fail_flag:
    #     shutil.rmtree(_frame_dir)
    #     shutil.rmtree(_face_dir)
    
    return not fail_flag
    
if __name__ == '__main__':
    import sys
    print()
    print('----------------Checking Start---------------- ')
    print()
    all_positive_clips = []
    with multiprocessing.Manager() as MG:

        transcripts_dir = './data/transcripts'
        video_clip_dir = './data/video_clips'
        audio_dir = './data/audio_clips'
        frame_dir = './data/frames'
        face_dir = './data/faces'
        comparE_tmp = './data/.tmp'
        feature_dir = './feature'

        # 流程
        extract_text = TranscriptExtractor(save_root=transcripts_dir)
        package_transcript = TranscriptPackager()
        cut_video = VideoCutterOneClip(save_root=video_clip_dir) # VideoCutter(save_root=video_clip_dir)
        filter_fun = FilterTranscrips(1)
        device = 0
        # 语音
        extract_audio = AudioSplitor(save_root=audio_dir)

        # 视觉
        # get_frames = Video2Frame(save_root=frame_dir)
        # get_faces = VideoFaceTracker(save_root=face_dir)
        # get_activate_spk = ActiveSpeakerSelector()
        # select_faces = FaceSelector()
        
        all_count = {
            'No_transcripts': []
        }
        all_movies = glob.glob('/data6/zjm/emobert/resources/raw_movies/*.mkv')
        all_movies += glob.glob('/data6/zjm/emobert/resources/raw_movies/*.mp4')
        for i, movie in enumerate(all_movies):
            print('[Main]: Processing', movie)
            basename = get_basename(movie)
            if basename not in ['No0001.The.Shawshank.Redemption', 'No0002.The.Godfather']:
                continue
        
            count = MG.dict({
                'no_sentence': [],
                'too_short': [],
                'is_not_en': [],
                'no_active_spk': [],
                'face_too_less': [] 
            })
            pool = multiprocessing.Pool(16)
            # 抽文本
            transcript_path = extract_text(movie)
            transcript_info = package_transcript(transcript_path)
            if transcript_info == None:
                all_count['No_transcripts'].append(movie)
                continue
            
            sentences = list(map(lambda  x: x['content'], transcript_info))
            # 检查是否句子长度>1
            _have_sentence = []
            for i, transcript in enumerate(transcript_info):
                if len(transcript['content']) > 0:
                    _have_sentence.append(transcript)
                else:
                    count['no_sentence'].append(transcripts['index'])

            # _have_sentence = list(filter(lambda x: len(x['content']) > 0, transcript_info))
            # count['no_sentence'] += [x for x in sentences and x not in _have_sentence]
            transcript_info = _have_sentence

            # 检查是否是英文
            # sentence_language = pool.map(detect_language, sentences)
            # eng_sentence = list(filter(lambda x: x[1]=='en', zip(transcript_info, sentence_language)))
            # transcript_info = list(map(lambda x: x[0], eng_sentence))
            # count['is_not_en'] += len(sentence_language) - len(eng_sentence)
            
            # 过滤太短的视频
            print('[Main]: Start filtering Transcipts')
            is_long_enough = list(tqdm(pool.imap(filter_fun, transcript_info), total=len(transcript_info)))
            transcript_info = list(filter(lambda x: x[1], zip(transcript_info, is_long_enough)))
            #len(is_long_enough) - len(transcript_info)
            count['too_short'] += [x for x,y in transcript_info if not y] 
            transcript_info = list(map(lambda x: x[0], transcript_info))
            print('[Main]: Remaining Transcipts: {}'.format(len(transcript_info)))

            # 切视频
            print('[Main]: Start cutting video')
            video_clip_dir = list(tqdm(
                pool.istarmap(cut_video, [(movie, transcript) for transcript in transcript_info]), 
                    total=len(transcript_info)
                ))[0]
            all_video_clip = sorted(glob.glob(os.path.join(video_clip_dir, '*.mkv')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            print('[Main]: Total clips found:', len(all_video_clip))
            # 切语音:
            print('[Main]: Start extracting audio files')
            audio_paths = list(tqdm(pool.imap(extract_audio, all_video_clip), total=len(all_video_clip)))
            print('[Main]: Total wav found:', len(audio_paths))
            # 抽帧抽脸
            print('[Main]: start vision process:')
            input_param = list(zip(all_video_clip, audio_paths))
            vision_result = list(tqdm(pool.imap(vision_process, input_param), total=len(input_param)))
            # vision_rvision_process(input_param[0])
            print('[Main]: totally {} clips passed vision test'.format(len(vision_result)))
            positive_clips = list(filter(lambda  x: x[1], zip(all_video_clip, vision_result)))
            positive_clips = list(map(lambda x: x[0], positive_clips))
            all_positive_clips += positive_clips
            pool.close()
            pool.join()
            positive_clips = list(map(lambda x: '/'.join(x.split('/')[-2:]), positive_clips))
            positive_clips = list(map(lambda x: x[:x.rfind('.')] + '\n', positive_clips))
            movie_name = get_basename(movie)
            mkdir("./data/check_data")
            save_path = os.path.join('./data/check_data', movie_name + '.txt')
            with open(save_path, 'w') as f:
                f.writelines(positive_clips)
            all_count[movie_name] = dict(count)
            json_path = os.path.join('./data/check_data', movie_name + '.json')
            json.dump(dict(count), open(json_path, 'w'), indent=4)

    # print(all_positive_clips)
    all_positive_clips = list(map(lambda x: '/'.join(x.split('/')[-2:]), all_positive_clips))
    all_positive_clips = list(map(lambda x: x[:x.rfind('.')]+'\n', all_positive_clips))
    print("Positive clips: {}".format(len(all_positive_clips)))
    with open('all_positive_clips.txt', 'w') as f:
        f.writelines(all_positive_clips)
    
    json_path = "negative_count.json"

    for key, value in all_count.items():
        if isinstance(value, multiprocessing.managers.DictProxy):
            all_count[key] = dict(value)
    
    json.dump(dict(all_count), open(json_path, 'w'), indent=4)