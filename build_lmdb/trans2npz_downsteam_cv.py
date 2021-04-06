import numpy as np
import h5py
import os
import json

'''
export PYTHONPATH=/data7/MEmoBert
将视觉信息处理成npz的格式, 
然后可以共同维护一个 img_db, 通过不同的 txt_db 进行索引.
这里的soft-label都是采用denseface的结构，FER+ 结构，所以是 8 类别
对于IEMOCAP 的raw-img 数据，在保存为h5的时候已经转化过了，所以可以直接保存。
'''

do_raw_img = True
corpus_name = 'IEMOCAP'
root_dir = f'/data7/emobert/exp/evaluation/{corpus_name}/'
cls_num = 8
if not do_raw_img:
    feature_dir = root_dir + 'feature/denseface_openface_msp_mean_std_torch'
    IMD_DIM=342
    ft_key = 'feat' # or trans1 trans2 or img_data
    npyz_dir = feature_dir + '/ft_npzs/fc'
else:
    feature_dir = root_dir + 'feature/openface_iemocap_raw_img'
    IMD_DIM=[112, 112]
    ft_key = 'img'
    npyz_dir = feature_dir + '/raw_img_npzs'

if not os.path.exists(npyz_dir):
    os.makedirs(npyz_dir)

h5_fts_path = feature_dir + '/1/{}.h5'
target_dir = f'/data7/emobert/exp/evaluation/{corpus_name}/refs/1'
target_path = os.path.join(target_dir, '{}_ref.json')
empty_video = 0
total_video = 0

'''
ref fromat:{'segmentId': {'label': 1}}
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
            soft_labels = np.zeros((1, cls_num), dtype=np.float)
            soft_labels[0][label] = 1.0
            if do_raw_img:
                feat = np.zeros([1] + IMD_DIM, dtype=np.float32)
            else:
                feat = np.zeros((1, IMD_DIM), dtype=np.float32)
        else:
            if do_raw_img:
                feat = np.array(data[segment_id]['img'])
                label = target[segment_id]['label']
                if isinstance(label, str):
                    label = int(label)
                soft_labels = np.zeros((len(feat), cls_num), dtype=np.float32)
                soft_labels[:, label] = 1.0
            else:
                soft_labels = np.array(data[segment_id]['pred'])
                feat = np.array(data[segment_id]['feat'])
        assert len(soft_labels) == len(feat)
        np.savez_compressed(outputfile,
                            soft_labels=soft_labels.astype(np.float16),
                            features=feat.astype(np.float16))       
        print('{} feat {}'.format(setname, feat.shape))
print('total {} and empty {}'.format(total_video, empty_video))