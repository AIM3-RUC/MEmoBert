import os
import h5py
import numpy as np
from copy import deepcopy
import sys
sys.path.append('/data7/MEmoBert/preprocess/tasks')
import pandas as pd
from functools import partial
from tqdm import tqdm
from preprocess.tasks.audio import ComParEExtractor
from preprocess.MELD.extract_denseface_comparE import extract_comparE_file, extract_features_h5

def get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/MSP/target'):
    utt_ids = np.load(os.path.join(target_root, f'all_int2name.npy'))
    return utt_ids

if __name__ == '__main__':
    
    output_dir = '/data7/emobert/exp/evaluation/MSP/feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if True:
        # for speech comparE
        audio_feature_dir = os.path.join(output_dir, 'comparE_raw')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        # comparE_model = ComParEExtractor(tmp_dir=f'{output_dir}/raw_fts')
        # # 偏函数: 主要目的是冻结固定参数？将所作用的函数作为 partial() 函数的第一个参数，原函数的各个参数依次作为 partial（）函数的后续参数，
        # # 原函数有关键字参数的一定要带上关键字，没有的话，按原有参数顺序进行补充.
        # extract_comparE = partial(extract_comparE_file, extractor_model=comparE_model)
        # save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        # audio_dir = '/data7/MEmoBert/emobert/exp/evaluation/MSP/audio'
        # utt_ids = get_all_utt_ids(target_root='/data7/MEmoBert/emobert/exp/evaluation/MSP/target')
        # print('total {} uttids'.format(len(utt_ids)))
        # extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)

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