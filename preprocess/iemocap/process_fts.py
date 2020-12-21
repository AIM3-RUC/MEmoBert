import numpy as np
import h5py
import os
import sys

'''
将视觉信息处理成npz的格式, 
然后可以共同维护一个 img_db, 通过不同的 txt_db 进行索引.
'''

if __name__ == "__main__":
    root_dir = '/data7/emobert/exp/evaluation/IEMOCAP/'
    feature_dir = root_dir + 'feature/denseface_openface_mean_std_movie_no_mask/'
    npyz_dir = feature_dir + 'ft_npzs'
    h5_fts_path = feature_dir + 'all.h5'

    if False:
        data = h5py.File(h5_fts_path)
        segment_ids = list(data.keys())
        print('total {} segments'.format(len(segment_ids)))

        for segment_id in segment_ids:
            outputfilename = segment_id + ".npz"
            outputfile = os.path.join(npyz_dir, outputfilename)
            if os.path.exists(outputfile):
                continue
            soft_labels = np.array(data[segment_id]['pred'])
            feat = np.array(data[segment_id]['feat'])
            assert len(soft_labels) == len(feat)
            np.savez_compressed(outputfile,
                                soft_labels=soft_labels.astype(np.float16),
                                features=feat.astype(np.float16))
    
    if False:
        set_name = 'trn'
        cv_no = sys.argv[1]
        names = []
        ft_path = os.path.join(root_dir, feature_dir, str(cv_no),  '{}.h5'.format(set_name))
        int2name_dir = os.path.join(root_dir, feature_dir, str(cv_no),  'public_split')
        if not os.path.exists(int2name_dir):
            os.makedirs(int2name_dir)
        data = h5py.File(ft_path)
        for key in data.keys():
            names.append(key)
        print('cv {} set {} {}'.format(cv_no, set_name, len(names)))
        int2name_path = os.path.join(int2name_dir, '{}_names.npy'.format(set_name))
        np.save(int2name_path, names)
