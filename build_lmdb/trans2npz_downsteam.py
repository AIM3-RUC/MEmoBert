import numpy as np
import h5py
import os
import sys
import json

'''
将视觉信息处理成npz的格式, 
然后可以共同维护一个 img_db, 通过不同的 txt_db 进行索引.
'''

root_dir = '/data7/emobert/exp/evaluation/MSP-IMPROV/'
feature_dir = root_dir + 'feature/denseface_seetaface_msp_mean_std/'
npyz_dir = feature_dir + '/ft_npzs/msp'
img_db_dir = feature_dir + '/img_db'
if not os.path.exists(npyz_dir):
    os.makedirs(npyz_dir)
if not os.path.exists(img_db_dir):
    os.makedirs(img_db_dir)

h5_fts_path = feature_dir + '/1/{}.h5'
target_path = '/data7/emobert/exp/evaluation/MSP-IMPROV/refs/1/{}_ref.json'
empty_video = 0
total_video = 0

'''
ref fromat:
{
    'segmentId': {'label': 1}
}
'''

for setname in ['trn', 'val', 'tst']:
    target = json.load(open(target_path.format(setname)))
    data = h5py.File(h5_fts_path.format(setname))
    segment_ids = list(data.keys())
    print('{} total {} segments'.format(setname, len(segment_ids)))
    for i, segment_id in enumerate(segment_ids):
        outputfilename = segment_id + ".npz"
        outputfile = os.path.join(npyz_dir, outputfilename)
        total_video += 1
        if os.path.exists(outputfile):
            continue
        if len(data[segment_id]) == 0:
            # 如果里面视觉信息是空的, soft-label 则是 ground-truth
            empty_video += 1
            label = target[segment_id]['label']
            if isinstance(label, str):
                label = int(label)
            soft_labels = np.zeros((1, 4), dtype=np.float)
            soft_labels[0][label] = 1.0
            feat = np.zeros((1, 342), dtype=np.float32)
        else:
            soft_labels = np.array(data[segment_id]['pred'])
            feat = np.array(data[segment_id]['feat'])
        assert len(soft_labels) == len(feat)
        np.savez_compressed(outputfile,
                            soft_labels=soft_labels.astype(np.float16),
                            features=feat.astype(np.float16))
print('total {} and empty {}'.format(total_video, empty_video))