import os, glob
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
from functools import partial
from toolz.sandbox import unzip
from preprocess.utils import get_basename, mkdir
from preprocess.tasks.audio import ComParEExtractor, Wav2VecExtractor
from preprocess.tasks.vision import DensefaceExtractor, FaceSelector
from preprocess.tasks.text import *
from preprocess.tools.get_emo_words import EmoLexicon
import preprocess.process_config as path_config

def extract_features_h5(extract_func, get_input_func, utt_ids, save_path):
    if os.path.exists(save_path):
        try:
            _h5f = h5py.File(save_path, 'r')
            if len(_h5f.keys()) == len(utt_ids):
                _h5f.close()
                return
            _h5f.close()
        except OSError:
            os.remove(save_path)

    h5f = h5py.File(save_path, 'w')

    for utt_id in tqdm(utt_ids):
        # utt_id = ''
        input_param = get_input_func(utt_id)
        feature = extract_func(input_param)
        if isinstance(feature, dict):
            utt_data = h5f.create_group(utt_id)
            for k, v in feature.items():
                utt_data[k] = v
        else:
            h5f[utt_id] = feature
    h5f.close()

def get_utt_id_files(meta_dir, file_name, moive_names_path):
    '''
    return: files with ['/data7/emobert/data_nomask_new/meta/No0079.The.Kings.Speech/has_active_spk.txt', 
        ...,]
    '''
    valid_movies_names = np.load(moive_names_path)
    files = []
    movie_names = []
    for name in valid_movies_names:
        filepath = os.path.join(meta_dir, name, f'{file_name}.txt')
        if os.path.exists(filepath):
            files.append(filepath)
            movie_names.append(name)
        else:
            print(f'[Warning] {name} is not exists')
    assert len(movie_names) == len(files)
    print(f'[INFO] {len(valid_movies_names)} {len(movie_names)} movies need to process')
    return files, movie_names

def process_emo_word(transcript_dir, utt_ids, emol, save_path, multiprocessing=False):
    if os.path.exists(save_path):
        content = json.load(open(save_path))
        if len(content.keys()) == len(utt_ids):
            return 
    
    json_path = utt_ids[0].split('/')[0]
    json_path = os.path.join(transcript_dir, json_path + '.json')
    utts = json.load(open(json_path))
    if not multiprocessing:
        all_word2affect = {}
        for utt_id in tqdm(utt_ids):
            utterance = utts[utt_id.split('/')[-1]]['content']
            emo_words, word2affect = emol.get_emo_words(utterance)
            all_word2affect[utt_id] = word2affect
    else:
        pool = multiprocessing.Pool(4)
        ret = pool.map(emol.get_emo_words, utt_ids)
        all_word2affect = list(map(lambda x:x[1], ret))
        all_word2affect = dict([(utt_id, word2affect) for utt_id, word2affect in zip(utt_ids, all_word2affect)])

    json.dump(all_word2affect, open(save_path, 'w'), indent=4)

def extract_denseface_trans_dir(dir_path, denseface_model, face_selector):
    # for one video clip
    active_spk = open(os.path.join(dir_path, 'has_active_spk.txt')).read().strip()
    assert active_spk != "None", dir_path
    active_spk = int(active_spk)
    infos = face_selector(dir_path, active_spk)
    imgs = [x['img'] for x in infos]
    frame_nums = [x['frame_num'] for x in infos]
    confidence = [x['confidence'] for x in infos]
    feats, preds, trans1s, trans2s = [], [], [], []
    for img in imgs:
        feat, pred = denseface_model(img)
        feats.append(feat)
        preds.append(pred)
        trans1, trans2 = denseface_model.get_mid_layer_output()
        trans1 = F.avg_pool2d(trans1, kernel_size=32, stride=1).view(trans1.size(0), -1).detach().cpu().numpy()
        trans2 = F.avg_pool2d(trans2, kernel_size=16, stride=1).view(trans2.size(0), -1).detach().cpu().numpy()
        trans1s.append(trans1)
        trans2s.append(trans2)

    feats = np.concatenate(feats, axis=0)
    preds = np.concatenate(preds, axis=0)
    trans1s = np.concatenate(trans1s, axis=0)
    trans2s = np.concatenate(trans2s, axis=0)
    frame_nums = np.array(frame_nums)
    confidence = np.array(confidence)
    return {'feat': feats, 'pred': preds, 'trans1': trans1s, 'trans2': trans2s, 'frame_idx': frame_nums, 'confidence': confidence}

def extract_comparE_file(audio_path, extractor_model):
    # for one audio clip, audio_path = audio_dir + "No0079.The.Kings.Speech/2" 
    audio_path = audio_path + '.wav'
    # print(audio_path)
    feat = extractor_model(audio_path)
    frame_nums = np.array(len(feat))
    return {'feat': feat, 'frame_idx': frame_nums}

def extract_wav2vec_file(audio_path, extractor_model):
    # for one audio clip, audio_path = audio_dir + "No0079.The.Kings.Speech/2" 
    audio_path = audio_path + '.wav'
    # print(audio_path)
    feat = extractor_model(audio_path)
    frame_nums = np.array(len(feat))
    return {'feat': feat, 'frame_idx': frame_nums}

if __name__ == '__main__':
    import sys
    utt_file_name = sys.argv[1]
    part_no, total = eval(sys.argv[2]), eval(sys.argv[3])

    extact_face_features = False
    extact_audio_features = True
    audio_features_type = 'wav2vec'
    use_asr_based_model=True
    feature_audio_root = path_config.feature_audio_wav2vec_dir

    transcripts_dir = path_config.transcript_json_dir
    video_clip_dir = path_config.video_clip_dir
    frame_dir = path_config.frame_dir
    face_dir = path_config.face_dir
    audio_dir = path_config.audio_dir
    meta_dir = path_config.meta_root
    moive_names_path = path_config.moive_names_path
    feature_face_root = path_config.feature_face_dir
    tmp_dir = path_config.tmp_dir

    ## for extracting face features
    mean = 63.987095
    std = 43.00519
    denseface = DensefaceExtractor()
    denseface.register_midlayer_hook([
        "features.transition1.relu",
        "features.transition2.relu"
    ])
    face_selector = FaceSelector()
    extract_denseface = partial(extract_denseface_trans_dir, denseface_model=denseface, face_selector=face_selector)
    all_utt_files, movie_names = get_utt_id_files(meta_dir, utt_file_name, moive_names_path)

    ## for extracting audio features
    # 偏函数: 主要目的是冻结固定参数？将所作用的函数作为 partial() 函数的第一个参数，原函数的各个参数依次作为 partial()函数的后续参数，
    # 原函数有关键字参数的一定要带上关键字，没有的话，按原有参数顺序进行补充.
    if audio_features_type == 'comparE':
        comparE_model = ComParEExtractor()
        extract_comparE = partial(extract_comparE_file, extractor_model=comparE_model)
    elif audio_features_type == 'wav2vec':
        wav2vec_model = Wav2VecExtractor(downsample=-1, gpu=0, use_asr_based_model=use_asr_based_model)
        extract_wav2vec = partial(extract_wav2vec_file, extractor_model=wav2vec_model)

    length = len(all_utt_files)
    start = int(part_no * length / total)
    end = int((part_no + 1) * length / total)
    all_utt_files = all_utt_files[start: end]
    print('[Main]: all utt_id files found:', len(all_utt_files))
    print('-------------------------------------------------')
    for i, movie_name in enumerate(movie_names):
        print(f'[{i}]\t{movie_name}')
    print('-------------------------------------------------')
    print('[Main]: movies to be processed:')
    print('-------------------------------------------------')
    
    movie_names = movie_names[start: end]
    for i, movie_name in zip(range(start, end), movie_names):
        print(f'[{i}]\t{movie_name}')
    print('-------------------------------------------------')
    
    for utt_file, movie_name in zip(all_utt_files, movie_names):
        # utt_file: one movie file: xxx/movie_name/has_active_spk.txt
        print(f'[Main]: Process {movie_name}:')
        utt_ids = open(utt_file).readlines()
        # one utt_id = "No0079.The.Kings.Speech/2"
        utt_ids = list(map(lambda x: x.strip(), utt_ids))
        if len(utt_ids) == 0:
            continue
        
        if extact_face_features:
            print("[INFO] Extracing denseface features!")
            feature_dir = os.path.join(feature_face_root, movie_name)
            mkdir(feature_dir)
            save_path = os.path.join(feature_dir, f'{utt_file_name}_denseface_with_trans.h5')
            print(save_path)
            extract_features_h5(extract_denseface, lambda x: os.path.join(face_dir, x), 
                        utt_ids, save_path)

        if extact_audio_features and audio_features_type=='comparE':
            print("[INFO] Extracing ComparE Audio features!")
            audio_feature_dir = os.path.join(feature_audio_root, movie_name)
            mkdir(audio_feature_dir)
            save_path = os.path.join(audio_feature_dir, f'{utt_file_name}_comparE.h5')
            print(save_path)
            extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)

        ## for extracting ComparE feature 
        if extact_audio_features and audio_features_type=='wav2vec':
            print("[INFO] Extracing Wav2vec Audio features!")
            audio_feature_dir = os.path.join(feature_audio_root, movie_name)
            mkdir(audio_feature_dir)
            if use_asr_based_model:
                save_path = os.path.join(audio_feature_dir, f'{utt_file_name}_wav2vec_asr.h5')
            else:
                save_path = os.path.join(audio_feature_dir, f'{utt_file_name}_wav2vec.h5')
            print(save_path)
            extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)