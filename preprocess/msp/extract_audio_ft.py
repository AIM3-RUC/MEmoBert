import os
import h5py
import numpy as np
from copy import deepcopy
import sys
sys.path.append('/data7/MEmoBert/preprocess/tasks')
import pandas as pd
from functools import partial
from tqdm import tqdm
from preprocess.tasks.audio import ComParEExtractor, Wav2VecExtractor, RawWavExtractor, Wav2VecCNNExtractor
from preprocess.MELD.extract_denseface_comparE import extract_comparE_file, extract_features_h5, extract_wav2vec_file, extract_rawwav_file
from preprocess.iemocap.extract_feats import get_trn_val_tst, split_by_utt_id

def get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/MSP/target'):
    utt_ids = np.load(os.path.join(target_root, f'all_int2name.npy'))
    return utt_ids

def split_h5(all_h5, save_root='feature/denseface'):
    h5f = h5py.File(all_h5, 'r')
    for cv in range(1, 11):
        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trn_int2name, val_int2name, tst_int2name, _, _, _ = get_trn_val_tst(cv, target_root='/data7/emobert/exp/evaluation/MSP/target')
        split_by_utt_id(h5f, trn_int2name, os.path.join(save_dir, 'trn.h5'))
        split_by_utt_id(h5f, val_int2name, os.path.join(save_dir, 'val.h5'))
        split_by_utt_id(h5f, tst_int2name, os.path.join(save_dir, 'tst.h5'))
    h5f.close()

if __name__ == '__main__':
    
    output_dir = '/data7/emobert/exp/evaluation/MSP/feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if False:
        # for speech comparE
        audio_feature_dir = os.path.join(output_dir, 'comparE_raw')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        comparE_model = ComParEExtractor(tmp_dir=f'{output_dir}/raw_fts')
        # # 偏函数: 主要目的是冻结固定参数？将所作用的函数作为 partial() 函数的第一个参数，原函数的各个参数依次作为 partial（）函数的后续参数，
        # # 原函数有关键字参数的一定要带上关键字，没有的话，按原有参数顺序进行补充.
        extract_comparE = partial(extract_comparE_file, extractor_model=comparE_model)
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        audio_dir = '/data7/MEmoBert/emobert/exp/evaluation/MSP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/MEmoBert/emobert/exp/evaluation/MSP/target')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        new_save_path = os.path.join(audio_feature_dir, 'all.h5')
        data = h5py.File(save_path)
        out_h5f = h5py.File(new_save_path, 'w')
        for uttid in data.keys():
            new_uttid = uttid.replace('audio_', '')
            tgt = data[uttid]['feat']
            if isinstance(tgt, h5py._hl.dataset.Dataset):
                out_h5f[new_uttid] = deepcopy(tgt[()])
            elif isinstance(tgt, h5py._hl.group.Group):
                _group = out_h5f.create_group(new_uttid)
                for key in tgt.keys():
                    _group[key] = deepcopy(tgt[key][()])
        out_h5f.close()
    
    if False:
        # for speech wav2vec2.0 
        use_asr_based_model = False
        if use_asr_based_model:
            audio_feature_dir = os.path.join(output_dir, 'wav2vec_raw_asr')
        else:
            audio_feature_dir = os.path.join(output_dir, 'wav2vec_raw')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        wav2vec_model = Wav2VecExtractor(downsample=-1, gpu=0, use_asr_based_model=use_asr_based_model)
        extract_wav2vec = partial(extract_wav2vec_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/MSP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/MEmoBert/emobert/exp/evaluation/MSP/target')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
    
    if True:
        # for speech wav2vec
        print('Start to split')
        output_dir = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_raw'
        save_path = os.path.join(output_dir, 'all.h5')
        split_h5(save_path, save_root=output_dir)
    
    if False:
        output_dir = '/data7/emobert/exp/evaluation/MSP/feature'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # for speech 
        audio_feature_dir = os.path.join(output_dir, 'wav2vec_rawwav')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        model_path = '/data7/emobert/resources/pretrained/wav2vec_base'
        wav2vec_model = RawWavExtractor(model_path, max_seconds=8)
        extract_rawwav = partial(extract_rawwav_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/MSP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/MSP/target')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_rawwav, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)

    if False:
        output_dir = '/data7/emobert/exp/evaluation/MSP/feature'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # for speech wav2vec2.0
        audio_feature_dir = os.path.join(output_dir, 'wav2vec_cnn')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        wav2vec_model = Wav2VecCNNExtractor(gpu=0)
        extract_wav2vec = partial(extract_wav2vec_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/MSP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/MSP/target')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
        
        # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=4 python extract_audio_ft.py