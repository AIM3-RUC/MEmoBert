import sys
import h5py
import os
import numpy as np
from tqdm import tqdm
from preprocess.FileOps import read_file

'''
将h5文件中的特征都转化为npz文件, 只需要存储特征和人脸即可
export PYTHONPATH=/data7/MEmoBert

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
        ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_denseface_with_trans.h5')
        denseface_ft = h5py.File(ft_path, mode='r')
        segment_indexs = list(denseface_ft[movie_name].keys())
        if len(segment_indexs) != movie2utts[movie_name]:
            print(movie_name, len(segment_indexs), movie2utts[movie_name])
        for segment_index in segment_indexs:
            outputfilename = movie_name + '_' + segment_index + ".npz"
            outputfile = os.path.join(output_dir, outputfilename)
            if os.path.exists(outputfile):
                continue
            soft_labels = np.array(denseface_ft[movie_name][segment_index]['pred'])
            feat = np.array(denseface_ft[movie_name][segment_index]['feat'])
            confidence = np.array(denseface_ft[movie_name][segment_index]['confidence'])
            frame_indexs = np.array(denseface_ft[movie_name][segment_index]['frame_idx'])
            assert len(soft_labels) == len(feat)
            np.savez_compressed(outputfile,
                                frame_idxs=frame_indexs.astype(np.float16),
                                confidence=confidence.astype(np.float16),
                                soft_labels=soft_labels.astype(np.float16),
                                features=feat.astype(np.float16))

def convert_hdf5_to_npz_voxceleb2(hdf5_dir, output_dir, movie_names_path, start=None, end=None):
    '''
    segment_id = movie_name + '_' + segment_index
    '''
    valid_movie_names = np.load(movie_names_path)
    if start is None:
        start = 0
    if end is None:
        end = len(valid_movie_names)
    print('total valid movies {} and start {} end {}'.format(len(valid_movie_names), start, end))
    for movie_name in tqdm(valid_movie_names[start:end]):
        splits = movie_name.split('#')
        videoId_dir = os.path.join(hdf5_dir, splits[0], splits[1])
        for segment_id in os.listdir(videoId_dir):
            ft_path = os.path.join(videoId_dir, segment_id)
            denseface_ft = h5py.File(ft_path, mode='r')
            outputfilename = movie_name + '#' + segment_id.replace('h5', 'npz')
            outputfile = os.path.join(output_dir, outputfilename)
            if os.path.exists(outputfile):
                continue
            
            # 如果video没有任何元素，应该就是没有人脸
            if len(list(denseface_ft.keys())) == 0:
                continue
            soft_labels = np.array(denseface_ft['pred'])
            # 如果人脸数目小于等于1，那么直接过滤掉
            if len(soft_labels) < 2:
                continue
            feat = np.array(denseface_ft['feat'])
            confidence = np.array(denseface_ft['confidence'])
            frame_indexs = np.array(denseface_ft['frame_idx'])
            assert len(soft_labels) == len(feat)
            np.savez_compressed(outputfile,
                                frame_idxs=frame_indexs.astype(np.float16),
                                confidence=confidence.astype(np.float16),
                                soft_labels=soft_labels.astype(np.float16),
                                features=feat.astype(np.float16))

if __name__ == "__main__":
    start = int(sys.argv[1])  # 0
    end =  int(sys.argv[2]) # 100
    if False:
        # for movies data
        hdf5_dir = '/data7/emobert/denseface_feature_nomask_torch/movies_v3'
        meta_data_dir = '/data7/emobert/data_nomask_new/meta'
        movie_names_path = '/data7/emobert/data_nomask_new/movies_v3/movie_names.npy'
        npzs_dir = '/data7/emobert/ft_npzs_nomask/movies_v3/fc' 
        if not os.path.exists(npzs_dir):
            os.makedirs(npzs_dir)
        convert_hdf5_to_npz(hdf5_dir, npzs_dir, meta_data_dir, movie_names_path, start=start, end=end)
    
    if True:
        ## for voxcelebs voxceleb2_v1: 5w多
        hdf5_dir = '/data13/voxceleb2/denseface_feature'
        movie_names_path = '/data7/emobert/data_nomask_new/voxceleb2_v1/movie_names.npy'
        npzs_dir = '/data7/emobert/ft_npzs_nomask/voxceleb2_v1/fc' 
        if not os.path.exists(npzs_dir):
            os.makedirs(npzs_dir)
        convert_hdf5_to_npz_voxceleb2(hdf5_dir, npzs_dir, movie_names_path, start=start, end=end)