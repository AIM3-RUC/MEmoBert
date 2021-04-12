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

def convert_hdf5_to_npz(hdf5_dir, output_dir, meta_data_dir, movie_names_path, start=None, end=None):
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
        ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_comparE.h5')
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
            # norm 
            norm_feat = (feat - mean) / std
            frame_indexs = np.array(list(range(0, int(np.array(audio_ft[movie_name][segment_index]['frame_idx'])))))
            assert len(norm_feat) == len(feat)
            np.savez_compressed(outputfile,
                                frame_idxs=frame_indexs.astype(np.float16),
                                features=norm_feat.astype(np.float16))

if __name__ == "__main__":
    start = int(sys.argv[1])  # 0
    end =  int(sys.argv[2]) # 100
    hdf5_dir = '/data7/emobert/comparE_feature/movies_v3'
    meta_data_dir = '/data7/emobert/data_nomask_new/meta'
    movie_names_path = '/data7/emobert/data_nomask_new/movies_v3/movie_names.npy'
    npzs_dir = '/data7/emobert/norm_comparE_npzs/movies_v3' 
    mean_std_path = '/data7/MEmoBert/emobert/comparE_feature/mean_std.npz'
    mean_std = np.load(mean_std_path, allow_pickle=True)
    mean = mean_std['mean']
    std = mean_std['std']
    # print(mean_std['mean'].shape)
    # print(mean_std['std'].shape)
    if not os.path.exists(npzs_dir):
        os.makedirs(npzs_dir)
    convert_hdf5_to_npz(hdf5_dir, npzs_dir, meta_data_dir, movie_names_path, start=start, end=end)