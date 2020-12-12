import sys
import h5py
import os
import numpy as np
from tqdm import tqdm
from preprocess.FileOps import read_file

'''

将h5文件中的特征都转化为npz文件, 只需要存储特征和人脸即可
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
        ft_path = os.path.join(hdf5_dir, movie_name, 'has_active_spk_denseface.h5')
        denseface_ft = h5py.File(ft_path)
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

if __name__ == "__main__":
    start = int(sys.argv[1])  # 0
    end =  int(sys.argv[2]) # 100
    npzs_dir = '/data7/emobert/ft_npzs/movies_v1' 
    hdf5_dir = '/data7/emobert/feature'
    meta_data_dir = '/data7/emobert/data_nomask/meta'
    movie_names_path = '/data7/emobert/data_nomask/movies_v1/movie_names.npy'

    if not os.path.exists(npzs_dir):
        os.makedirs(npzs_dir)
    convert_hdf5_to_npz(hdf5_dir, npzs_dir, meta_data_dir, movie_names_path, start=start, end=end)