import sys
import h5py
import os
import numpy as np
from numpy.core.fromnumeric import std
from tqdm import tqdm
from preprocess.FileOps import read_file

'''
export PYTHONPATH=/data7/MEmoBert

将h5文件中的特征都转化为npz文件, 需要计算所有的特征的均值和方差，然后保存的是 normalized comparE.
均值和方差的地址是:
'''

def convert_hdf5_to_npz(hdf5_dir, output_dir, mean, std, use_mean_pooling):
    '''
    '''
    ft_path = os.path.join(hdf5_dir, 'all.h5')
    audio_ft = h5py.File(ft_path, mode='r')
    # Ses05M_script03_2_M009
    segment_indexs = audio_ft.keys()
    for segment_index in segment_indexs:
        outputfilename = segment_index + ".npz"
        outputfile = os.path.join(output_dir, outputfilename)
        if os.path.exists(outputfile):
            continue
        feat = np.array(audio_ft[segment_index])
        # norm 
        norm_feat = (feat - mean) / std
        if len(norm_feat) == 0:
            print('segment {} norm {}'.format(segment_index, len(norm_feat)))
        assert len(norm_feat) == len(feat)
        if use_mean_pooling:
            mean_norm_feat = []
            for i in range(0, len(norm_feat), 5):
                if i+5 >= len(norm_feat):
                    mean_norm_feat.append(np.mean(norm_feat[i:], axis=0))
                else:
                    mean_norm_feat.append(np.mean(norm_feat[i:i+5], axis=0))
            if len(mean_norm_feat) == 0:
                print('[Afer Mean]segment {} meam-norm {}'.format(segment_index, len(mean_norm_feat)))
            norm_feat = np.array(mean_norm_feat)
        frame_indexs = np.arange(0, len(norm_feat))
        np.savez_compressed(outputfile,
                            frame_idxs=frame_indexs.astype(np.float16),
                            features=norm_feat.astype(np.float16))

def cal_mean_std(hdf5_dir, iemocap_mean_std_path):
    ft_path = os.path.join(hdf5_dir, 'all.h5')
    audio_ft = h5py.File(ft_path, mode='r')
    # Ses05M_script03_2_M009
    all_fts = []
    segment_indexs = audio_ft.keys()
    for segment_index in segment_indexs:
        if isinstance(audio_ft[segment_index], h5py._hl.group.Group):
            all_fts.append(np.array(audio_ft[segment_index]))
            print(np.array(audio_ft[segment_index]).shape, type(audio_ft[segment_index]))
        else:
            all_fts.append(audio_ft[segment_index])
    print('all_fts {}'.format(len(all_fts)))
    all_fts = np.concatenate(all_fts)
    print('all_fts {}'.format(len(all_fts)))
    mean = np.mean(all_fts, axis=0)
    std = np.std(all_fts, axis=0)
    print('mean, std {} {}'.format(mean.shape, std.shape))
    np.savez(
        iemocap_mean_std_path,
        mean=mean, 
        std = std
    )

if __name__ == "__main__":
    corpus_name = 'MELD'
    use_mean_pooling = True # 连续的5帧进行平均
    hdf5_dir = '/data7/emobert/exp/evaluation/{}/feature/comparE_raw'.format(corpus_name)

    if False:
        # computing the mean and std of all iemocap 
        iemocap_mean_std_path = '/data7/emobert/exp/evaluation/{}/feature/comparE_raw/mean_std.npz'.format(corpus_name)
        cal_mean_std(hdf5_dir, iemocap_mean_std_path)

    if True:
        mean_std_path = '/data7/MEmoBert/emobert/comparE_feature/mean_std.npz'
        if use_mean_pooling:
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/movies_norm_comparE_npzs_5mean'.format(corpus_name)
        else:
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/movies_norm_comparE_npzs'.format(corpus_name)
        mean_std = np.load(mean_std_path, allow_pickle=True)
        mean = mean_std['mean']
        std = mean_std['std']
        # print(mean_std['mean'].shape)
        # print(mean_std['std'].shape)
    else:
        self_mean_std_path = '/data7/emobert/exp/evaluation/{}/feature/comparE_raw/mean_std.npz'.format(corpus_name)
        if use_mean_pooling:
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/norm_comparE_npzs_5mean'.format(corpus_name)
        else:
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/norm_comparE_npzs'.format(corpus_name)
        mean_std = np.load(self_mean_std_path, allow_pickle=True)
        mean = mean_std['mean']
        std = mean_std['std']

    if True:
        if not os.path.exists(npzs_dir):
            os.makedirs(npzs_dir)
        convert_hdf5_to_npz(hdf5_dir, npzs_dir, mean, std, use_mean_pooling)