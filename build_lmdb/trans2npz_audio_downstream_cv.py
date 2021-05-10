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

def convert_hdf5_to_npz(hdf5_dir, output_dir, mean, std, use_mean_pooling, pooling_num_frames):
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

        if isinstance(audio_ft[segment_index], h5py._hl.dataset.Dataset):
            feat = np.array(audio_ft[segment_index])
        elif isinstance(audio_ft[segment_index], h5py._hl.group.Group):
            feat = np.array(audio_ft[segment_index]['feat'])
        else:
            print('[Error] type error')

        if len(feat.shape) == 3:
            print('feat shape {}'.format(feat.shape))

        if mean is not None and std is not None: 
            print('norm')
            feat = (feat - mean) / std

        if len(feat) == 0:
            print('segment {}'.format(segment_index))

        if use_mean_pooling:
            mean_norm_feat = []
            for i in range(0, len(feat), pooling_num_frames):
                if i+pooling_num_frames >= len(feat):
                    mean_norm_feat.append(np.mean(feat[i:], axis=0))
                else:
                    mean_norm_feat.append(np.mean(feat[i:i+pooling_num_frames], axis=0))
            if len(mean_norm_feat) == 0:
                print('[Afer Mean]segment {} meam-norm {}'.format(segment_index, len(mean_norm_feat)))
            feat = np.array(mean_norm_feat)
        frame_indexs = np.arange(0, len(feat))
        np.savez_compressed(outputfile,
                            frame_idxs=frame_indexs.astype(np.float16),
                            features=feat.astype(np.float16))

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
    corpus_name = 'MSP'
    use_mean_pooling = True # 连续的xx帧进行平均
    feat_type = 'rawwav'
    pooling_num_frames = 3
    use_asr_based_model = True

    if False:
        # computing the mean and std of all iemocap 
        iemocap_mean_std_path = '/data7/emobert/exp/evaluation/{}/feature/comparE_raw/mean_std.npz'.format(corpus_name)
        cal_mean_std(hdf5_dir, iemocap_mean_std_path)

    if feat_type == 'wav2vec':
        mean, std = None, None
        if use_asr_based_model:
            hdf5_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_raw_asr'.format(corpus_name)
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_asr_npzs_3mean'.format(corpus_name)
        else:
            hdf5_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_raw'.format(corpus_name)
            npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_npzs_3mean'.format(corpus_name)
    elif feat_type == 'rawwav':
        hdf5_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_rawwav'.format(corpus_name)
        npzs_dir = '/data7/emobert/exp/evaluation/{}/feature/wav2vec_rawwav_npzs'.format(corpus_name)
        use_mean_pooling = False
        mean, std = None, None
    else:
        hdf5_dir = '/data7/emobert/exp/evaluation/{}/feature/comparE_raw'.format(corpus_name)
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
        print(hdf5_dir)
        print(npzs_dir)
        if not os.path.exists(npzs_dir):
            os.makedirs(npzs_dir)
        convert_hdf5_to_npz(hdf5_dir, npzs_dir, mean, std, use_mean_pooling=use_mean_pooling, pooling_num_frames=pooling_num_frames)