import os, glob
import h5py
import json
import numpy as np
import multiprocessing
import gc
from tqdm import tqdm
from collections import Counter
from preprocess.tools import istarmap
from preprocess.utils import get_basename, mkdir
from preprocess.tasks.audio import AudioSplitor
from preprocess.tasks.vision import Video2Frame, VideoFaceTracker, ActiveSpeakerSelector, FaceSelector
from preprocess.tasks.text import TranscriptExtractor, TranscriptPackager
from preprocess.tasks.common import VideoCutterOneClip, FilterTranscrips, pool_filter
import preprocess.process_config as path_config

def save_h5(feature, lengths, save_path):
    h5f = h5py.File(save_path, 'w')
    h5f['feature'] = feature
    h5f['lengths'] = lengths
    h5f.close()

def save_uttid(uttids, save_path):
    f = open(save_path, 'w')
    f.writelines(list(map(lambda x: str(x) + '\n', uttids)))
    print(f'[Main]: save utt_id to {save_path}')

def has_face(utt_id):
    global face_dir
    clip_num = utt_id.split('/')[-1]
    _face_dir = os.path.join(face_dir, utt_id)
    ans = glob.glob(os.path.join(_face_dir, clip_num+'_aligned', '*.bmp'))
    return bool(ans)

def get_audio_path(utt_id):
    return os.path.join(audio_dir, utt_id + '.wav')

def filter_face(uttid):
    global face_dir
    global frame_dir
    clip_num = uttid.split('/')[-1]
    _face_dir = os.path.join(face_dir, uttid)
    _frame_dir = os.path.join(frame_dir, uttid) 
    frame_len = len(glob.glob(os.path.join(_frame_dir, '*.jpg')))
    face_pics = glob.glob(os.path.join(_face_dir, clip_num+'_aligned', '*.bmp'))
    face_frames = set(map(lambda x: int(x.split('.')[0].split('_')[-1], face_pics)))
    face_len = len(face_frames)
    return face_len >= 0.2 * frame_len

def filter_face_longest_spk(uttid):
    global face_dir
    global frame_dir
    clip_num = uttid.split('/')[-1]
    _face_dir = os.path.join(face_dir, uttid)
    _frame_dir = os.path.join(frame_dir, uttid) 
    frame_len = len(glob.glob(os.path.join(_frame_dir, '*.jpg')))
    face_pics = glob.glob(os.path.join(_face_dir, clip_num+'_aligned', '*.bmp'))
    spks = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-2]), face_pics))
    spks_count = Counter(spks)
    max_spk = sorted(spks_count.items(), key=lambda x: x[1], reverse=True)
    max_value = max_spk[0][1]
    return_value = None
    if max_value >= 0.2 * frame_len:
        return_value = True
    else:
        return_value = False
    
    for spk, face_num in max_spk:
        if face_num < 0.2 * frame_len:
            face_to_be_remove = glob.glob(os.path.join(_face_dir, clip_num+'_aligned', f'frame_det_{spk}_*.bmp'))
            for face in face_to_be_remove:
                os.remove(face)
    
    return return_value

def find_exists(movie_name):
    # if there has has_active_spk.txt, which denotes the movie is preprocess done.
    _dir = os.path.join(path_config.meta_root, movie_name)
    act_spk_file = os.path.join(_dir, 'has_active_spk.txt')
    return True if os.path.exists(act_spk_file) else False

def read_movie_names():
    path = '/data7/MEmoBert/preprocess/data/movies_v1/movie_names.npy'
    return np.load(path).tolist()

if __name__ == '__main__':
    # movie_indexs = list(range(1, 100))
    num_worker = 24
    chunk_size = 20
    print()
    print('----------------Preprocessing Start---------------- ')
    print('process movies:')
    for i, m in enumerate(movie_names):
        print(i, m)
    print('--------------------------------------------------- ')
    print('process {} and chunk size {}'.format(num_worker, chunk_size))
    
    all_positive_clips = []
    with multiprocessing.Manager() as MG:
        raw_movies_dir = path_config.raw_movies_dir
        transcripts_dir =  path_config.transcripts_dir
        transcript_json_dir = path_config.transcript_json_dir
        video_clip_dir = path_config.video_clip_dir
        audio_dir = path_config.audio_dir
        frame_dir = path_config.frame_dir
        face_dir = path_config.face_dir
        meta_root = path_config.meta_root
        check_data_dir= path_config.check_data_dir

        # 流程
        extract_text = TranscriptExtractor(save_root=transcripts_dir)
        package_transcript = TranscriptPackager(save_root=transcript_json_dir)
        cut_video = VideoCutterOneClip(save_root=video_clip_dir) # VideoCutter(save_root=video_clip_dir)
        filter_transcript = FilterTranscrips(threshold=1)
        device = 0

        # 语音
        extract_audio = AudioSplitor(save_root=audio_dir)

        # 视觉
        get_frames = Video2Frame(save_root=frame_dir)
        get_faces = VideoFaceTracker(openface_dir=path_config.openface_dir, save_root=face_dir)
        get_activate_spk = ActiveSpeakerSelector()
        select_faces = FaceSelector()

        all_count = {
            'No_transcripts': []
        }

        # for movie_index in movie_indexs:
        #     movies = []
        #     for _format in ['mkv', 'mp4', 'rmvb', 'avi', 'wmv', 'rm', 'ram']:
        #         movies += glob.glob('{}/No{:04d}*.{}'.format(raw_movies_dir, movie_index, _format))
        #     if len(movies) == 0:
        #         print('[Warning]: No {} movie index'.format(movie_index))
        #         continue
        #     assert len(movies) == 1, print(movies)
        #     movie = movies[0]
        
        movie_names = read_movie_names()
        for movie_name in movie_names:
            movies = []
            for _format in ['mp4', 'mkv', 'rmvb', 'avi', 'wmv', 'rm', 'ram']:
                path = os.path.join(raw_movies_dir, movie_name + '.' + _format)
                if os.path.exists(path):
                    movies.append(path)
                if len(movies) == 0:
                    print('[Warning]: No {} movie index'.format(movie_name))
                continue
            
            assert len(movies) >= 1, print(movies)
            movie = movies[0]
            print('[Main]: Processing', movie)
            movie_name = get_basename(movie)
            if find_exists(movie_name):
                print('[Main]: {} exists skip'.format(movie_name))
                continue
            meta_dir = os.path.join(meta_root, movie_name)
            mkdir(meta_dir)
            count = {
                'no_sentence': [],
                'too_short': [],
                'no_face': [],
                'face_too_less': [], 
                'no_active_spk': [],
            }
            # judge if there are transcripts info of this movie
            transcript_path = extract_text(movie)
            transcript_info = package_transcript(transcript_path)
            if transcript_info == None:
                all_count['No_transcripts'].append(movie)
                continue
        
            sentences = list(map(lambda  x: x['content'], transcript_info))
            # 检查是否句子长度>=1
            _have_sentence = []
            for i, transcript in enumerate(transcript_info):
                if len(transcript['content']) > 0:
                    _have_sentence.append(transcript)
                else:
                    count['no_sentence'].append(transcript['index'])

            transcript_info = _have_sentence

            pool = multiprocessing.Pool(num_worker)
            # 过滤时间小于1s的句子
            print('[Main]: Start filtering Transcipts')
            is_long_enough = list(tqdm(pool.imap(filter_transcript, transcript_info, chunksize=chunk_size), total=len(transcript_info)))
            transcript_info = list(filter(lambda x: x[1], zip(transcript_info, is_long_enough)))
            count['too_short'] += [x['index'] for x,y in transcript_info if not y] 
            transcript_info = list(map(lambda x: x[0], transcript_info))
            print('[Main]: Remaining Transcipts: {}'.format(len(transcript_info)))
            
            utt_ids = list(map(lambda x: movie_name + '/' + str(x['index']), transcript_info))
            save_uttid(utt_ids, os.path.join(meta_dir, 'base.txt'))
            
            if len(transcript_info) == 0:
                print("There is not avalible transcripts!!!")
                continue
            
            # 切视频
            print('[Main]: Start cutting video')
            video_clip_dir = list(tqdm(
                pool.istarmap(cut_video, [(movie, transcript) for transcript in transcript_info]), 
                    total=len(transcript_info)
                ))[0]
            
            all_video_clip = sorted(glob.glob(os.path.join(video_clip_dir, '*.mp4')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            print('[Main]: Total clips found:', len(all_video_clip))

            # 切语音:
            print('[Main]: Start extracting audio files')
            audio_paths = list(tqdm(pool.imap(extract_audio, all_video_clip, chunksize=chunk_size), total=len(all_video_clip)))
            print('[Main]: Total wav found:', len(audio_paths))

            # 抽帧抽脸 
            print('[Main]: Start extracting frames')
            frame_dirs = list(tqdm(pool.imap(get_frames, all_video_clip, chunksize=chunk_size), total=len(all_video_clip)))
            print('[Main]: Start extracting faces') 
            # modify to imap imap_unordered and increase the chunk-size
            face_dirs = list(tqdm(pool.imap_unordered(get_faces, frame_dirs, chunksize=chunk_size), total=len(frame_dirs)))
            # 先去除完全没有人脸的片段, 保存成一个json
            print('[Main]: Filtering out clip with no faces')
            _utt_ids = pool_filter(has_face, utt_ids, pool)
            count['no_face'] += [int(x.split('/')[-1]) for x in utt_ids if x not in _utt_ids] 
            utt_ids = _utt_ids
            save_uttid(utt_ids, os.path.join(meta_dir, 'has_face.txt'))
            print('[Main]: Total clips found with face:', len(utt_ids))

            # 出现时间最长说话人占视频的20%(任何人脸) 保存成一个json
            print('[Main]: Filtering out clip whose spk is shoter than 0.2 frame length:')
            _utt_ids = pool_filter(filter_face_longest_spk, utt_ids, pool)
            count['face_too_less'] += [int(x.split('/')[-1]) for x in utt_ids if x not in _utt_ids] 
            utt_ids = _utt_ids
            save_uttid(utt_ids, os.path.join(meta_dir, 'longest_spk_0.2.txt'))
            print('[Main]: Total clips found with longest face > 0.2*frames:', len(utt_ids))

            # 删掉小于出现时长小于总时长20%的人脸再做检测，看是否有说话人
            print('[Main]: Active speaker detection')
            active_spks = list(tqdm(
                pool.istarmap(get_activate_spk, 
                        [(os.path.join(face_dir, x), get_audio_path(x)) for x in utt_ids], chunksize=chunk_size), 
                        total=len(all_video_clip)
            ))

            pool.close()
            pool.join()

            uttid_actspk = list(zip(utt_ids, active_spks))
            _utt_ids = list(map(lambda x: x[0], filter(lambda x: x[1] is not None, uttid_actspk)))
            count['no_active_spk'] += [int(x.split('/')[-1]) for x in utt_ids if x not in _utt_ids] 
            utt_ids = _utt_ids
            save_uttid(utt_ids, os.path.join(meta_dir, 'has_active_spk.txt'))
            print('[Main]: Total clips found with active speaker:', len(utt_ids))

            # 统计数据
            mkdir(check_data_dir)
            save_path = os.path.join(check_data_dir, movie_name + '.txt')
            all_count[movie_name] = dict(count)
            json_path = os.path.join(check_data_dir, movie_name + '.json')
            json.dump(dict(count), open(json_path, 'w'), indent=4)
            gc.collect()
            print('-------------------- Done and Next One --------------------------------\n')

    # # dump all negative count
    # json_path = os.path.join(check_data_dir, "/negative_count.json")
    # for key, value in all_count.items():
    #     if isinstance(value, multiprocessing.managers.DictProxy):
    #         all_count[key] = dict(value)
    # json.dump(dict(all_count), open(json_path, 'w'), indent=4)

    # PYTHONPATH=/data7/MEmoBert/ python process.py 