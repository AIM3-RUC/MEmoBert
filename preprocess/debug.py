import os, glob
import h5py
import numpy as np
import json
from tqdm import tqdm
from functools import partial
from toolz.sandbox import unzip
from preprocess.utils import get_basename, mkdir
from preprocess.tasks.audio import *
from preprocess.tasks.vision import *
from preprocess.tasks.text import *
from preprocess.tools.get_emo_words import EmoLexicon
import preprocess.process_config as path_config

def extract_features_h5(extract_func, get_input_func, utt_ids, save_path, multi_processing=False):
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
        input_param = get_input_func(utt_id)
        feature = extract_func(input_param)
        if isinstance(feature, dict):
            utt_data = h5f.create_group(utt_id)
            for k, v in feature.items():
                utt_data[k] = v
        else:
            h5f[utt_id] = feature
    h5f.close()

def get_utt_id_files(meta_dir, file_name):
    files = glob.glob(os.path.join(meta_dir, f'*/{file_name}.txt'))
    files = sorted(files)
    movie_names = list(map(lambda x: x.split('/')[-2], files))
    return files, movie_names

def process_emo_word(utt_ids, emol, save_path, multiprocessing=False):
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
        utterances = list(map(lambda x: utts[x.split('/')[-1]]['content'], utt_ids))
        ret = pool.map(emol.get_emo_words, utt_ids)
        all_word2affect = list(map(lambda x:x[1], ret))
        all_word2affect = dict([(utt_id, word2affect) for utt_id, word2affect in zip(utt_ids, all_word2affect)])

    json.dump(all_word2affect, open(save_path, 'w'), indent=4)

def extract_denseface_dir(dir_path, denseface_model, face_selector):
    active_spk = open(os.path.join(dir_path, 'activate_spk.txt')).read().strip()
    assert active_spk != "None", dir_path
    active_spk = int(active_spk)
    imgs = face_selector(dir_path, active_spk)
    feats, pred = map(list, unzip(filter(lambda x: x, [denseface_model(x) for x in imgs])))
    feats = np.concatenate(feats, axis=0)
    pred = np.concatenate(pred, axis=0)
    return {'feat': feats, 'pred': pred}

if __name__ == '__main__':
    device = 0
    transcripts_dir = path_config.transcript_json_dir
    video_clip_dir = path_config.video_clip_dir
    audio_dir = path_config.audio_dir
    frame_dir = path_config.frame_dir
    face_dir = path_config.face_dir
    meta_dir = path_config.meta_root
    feature_root = path_config.feature_dir
    tmp_dir = path_config.tmp_dir
        
    denseface = DensefaceExtractor(device=device, mean=48.85351, std=45.574123)
    face_selector = FaceSelector()
    extract_denseface = partial(extract_denseface_dir, denseface_model=denseface, face_selector=face_selector)

    movie_name = 'No0103.Home.Alone'
    utt_file = os.path.join(meta_dir, movie_name, 'has_active_spk.txt')
    utt_file_name = 'has_active_spk'
    feature_dir = os.path.join(feature_root, movie_name)
    mkdir(feature_dir)
    utt_ids = open(utt_file).readlines()
    utt_ids = list(map(lambda x: x.strip(), utt_ids))
    print(len(utt_ids))
    save_path = os.path.join(feature_dir, f'{utt_file_name}_denseface.h5')
    extract_features_h5(extract_denseface, lambda x: os.path.join(face_dir, x), 
                utt_ids, save_path)