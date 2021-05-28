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

def convert_hdf5_to_npz(hdf5_dir, output_dir, meta_data_dir, movie_names_path, use_mean_pooling=False, pooling_num_frames=5, start=None, end=None):
    '''
    fileter_dict: 未来加很多数据的时候可能
    segment_id = movie_name + '_' + segment_index
    No0123.Brokeback.Mountain_1.npz
    '''
    valid_movie_names = np.load(movie_names_path)
    movie2utts ={}
    for movie_name in valid_movie_names:
        # verify the 
        active_spk_filepath = os.path.join(meta_data_dir, movie_name, 'has_active_spk.txt')
        active_spk_lines = read_file(active_spk_filepath)
        movie2utts[movie_name] = len(active_spk_lines)
    if start is None:
        start = 0
    if end is None:
        end = len(valid_movie_names)
    print('total valid movies {} and start {} end {}'.format(len(valid_movie_names), start, end))
    for movie_name in tqdm(valid_movie_names[start:end]):
        if feat_type == 'comparE':
            ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_comparE.h5')
        elif feat_type == 'rawwav':
            ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_rawwav.h5')
        else:
            if use_asr_based_model:
                ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_wav2vec_asr.h5')
            else:
                ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_wav2vec.h5')
        audio_ft = h5py.File(ft_path, mode='r')
        segment_indexs = list(audio_ft[movie_name].keys())
        if len(segment_indexs) != movie2utts[movie_name]:
            print(movie_name, len(segment_indexs), movie2utts[movie_name])
        for segment_index in segment_indexs:
            outputfilename = movie_name + '_' + segment_index + ".npz"
            outputfile = os.path.join(output_dir, outputfilename)
            if os.path.exists(outputfile):
                continue
            feat = np.array(audio_ft[movie_name][segment_index]['feat'])

            if feat_type == 'wav2vec':
                # dim-0 is batchsize=1
                feat = feat[0]
            
            if len(feat) == 0:
                print('segment {} have no frames'.format(segment_index))
            if mean is not None and std is not None:
                print('with mean-std norm')
                feat = (feat - mean) / std

            if use_mean_pooling:
                # 连续的 pooling_num_frames=5 帧进行平均, 360 / 5 = 90, (300,130) -> (300/5, 5, 130)
                mean_norm_feat = []
                for i in range(0, len(feat), pooling_num_frames):
                    if i+pooling_num_frames >= len(feat):
                        mean_norm_feat.append(np.mean(feat[i:], axis=0))
                    else:
                        mean_norm_feat.append(np.mean(feat[i:i+pooling_num_frames], axis=0))
                if len(mean_norm_feat) == 0:
                    print('[Afer Mean]segment {} meam-norm {}'.format(segment_index, len(mean_norm_feat)))
                feat = np.array(mean_norm_feat)
            frame_indexs = np.array(list(range(0, len(feat))))
            np.savez_compressed(outputfile,
                                frame_idxs=frame_indexs.astype(np.float16),
                                features=feat.astype(np.float16))

if __name__ == "__main__":
    start = int(sys.argv[1])  # 0
    end =  int(sys.argv[2]) # 100
    feat_type = 'rawwav'
    if feat_type == 'comaprE':
        # 10ms/frame
        use_mean_pooling = True # 连续的5帧进行平均
        pooling_num_frames = 5
        hdf5_dir = '/data7/emobert/comparE_feature/movies_v2'
        meta_data_dir = '/data7/emobert/data_nomask_new/meta'
        movie_names_path = '/data7/emobert/data_nomask_new/movies_v2/movie_names.npy'
        npzs_dir = '/data7/emobert/norm_comparE_npzs/movies_v2_5mean' 
        mean_std_path = '/data7/MEmoBert/emobert/comparE_feature/mean_std.npz'
        mean_std = np.load(mean_std_path, allow_pickle=True)
        mean = mean_std['mean']
        std = mean_std['std']
        # print(mean_std['mean'].shape)
        # print(mean_std['std'].shape)
    elif feat_type == 'wav2vec':
        # 20ms/frame
        use_mean_pooling = True # 连续的3帧进行平均
        use_asr_based_model = True
        pooling_num_frames = 3
        hdf5_dir = '/data7/emobert/wav2vec_feature/movies_v1'
        meta_data_dir = '/data7/emobert/data_nomask_new/meta'
        movie_names_path = '/data7/emobert/data_nomask_new/movies_v1/movie_names.npy'
        if use_asr_based_model:
            npzs_dir = '/data7/emobert/wav2vec_feature_npzs/movies_v1_asr_3mean' 
        else:
            npzs_dir = '/data7/emobert/wav2vec_feature_npzs/movies_v1_3mean' 
        mean, std = None, None
    else:
        hdf5_dir = '/data7/emobert/wav2vec_feature/movies_v3'
        meta_data_dir = '/data7/emobert/data_nomask_new/meta'
        movie_names_path = '/data7/emobert/data_nomask_new/movies_v3/movie_names.npy'
        npzs_dir = '/data7/emobert/wav2vec_feature_npzs/movies_v3_rawwav' 
        mean, std = None, None
        use_mean_pooling = False
        pooling_num_frames = None

    if not os.path.exists(npzs_dir):
        os.makedirs(npzs_dir)
    convert_hdf5_to_npz(hdf5_dir, npzs_dir, meta_data_dir, movie_names_path, use_mean_pooling, pooling_num_frames, start=start, end=end)