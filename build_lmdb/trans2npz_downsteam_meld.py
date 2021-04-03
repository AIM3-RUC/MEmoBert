import numpy as np
import h5py
import os
import sys
import json
from build_lmdb.mk_rawimg_db import prepocess_img, prepocess_img_data

'''
export PYTHONPATH=/data7/MEmoBert

将视觉信息处理成npz的格式, 
然后可以共同维护一个 img_db, 通过不同的 txt_db 进行索引.
对于MELD数据来说，由于也是多人的场景，如果很多视频可能无法追踪到说话的那个人，效果比较差。
a['val']['dia0_utt0']['frames_idx'][1] = 2002, 表示的是第2个人的第2张脸。
a['val']['dia0_utt0']['img_data'][1] = (112, 112, 3)
如果有 activate spk, 那么使用该 activate spk 的人脸。
如果没有 activate spk, 那么使用人脸最多的那个人的人脸序列。
统计平均的人脸数目，以及有多少 videos 没有人脸存在。
注意存储的数据格式：对于frame-idx, 如果有 active_spk_id 那么存储形式是该 spk 的frames, 比如 1 2 3 等
    如果没有 active_spk_id 那么存储形式是该 所有spk的frames, 比如 1001 2001 3001 等
保存原始图片数据的策略跟特征保持一致～。
'''

do_raw_img = True
corpus_name = 'MELD'
num_cls = 8 # 用于 MRC-KL 任务，所以跟denseface训练的保持一致，是8类
root_dir = f'/data7/emobert/exp/evaluation/{corpus_name}/'
feature_dir = root_dir + 'feature/denseface_openface_meld_mean_std_torch/'
if not do_raw_img:
    IMD_DIM=342
    ft_key = 'feat' # or trans1 trans2 or img_data
    npyz_dir = feature_dir + '/ft_npzs/fc'
else:
    feature_dir = root_dir + 'feature/denseface_openface_meld_mean_std_torch/'
    IMD_DIM=[112, 112, 3]
    img_size = 112
    mean, std = 67.61417, 37.89171
    ft_key = 'img_data'
    npyz_dir = feature_dir + '/raw_img_npzs'

if not os.path.exists(npyz_dir):
    os.makedirs(npyz_dir)

h5_fts_path = feature_dir + '{}.h5'
target_dir = f'/data7/emobert/exp/evaluation/{corpus_name}/refs'
target_path = os.path.join(target_dir, '{}.json')

'''
ref fromat:{'segmentId': {'label': 1}}
'''

def empty_data_process(target, segment_id, do_raw_img):
    # 如果里面视觉信息是空的, soft-label 则是 ground-truth
    # print('[Debug] No visual info')
    if do_raw_img:
         # 如果使用原始的数据，那么不需要soft_labels, batchsize=1
        feat = np.zeros([1] + IMD_DIM)
        confidence = np.array([1.0])
        return None, feat, confidence
    else:
        label = target[segment_id]['label']
        if isinstance(label, str):
            label = int(label)
        soft_labels = np.zeros((1, num_cls), dtype=np.float)
        soft_labels[0][label] = 1.0
        feat = np.zeros((1, IMD_DIM), dtype=np.float32)
        confidence = np.array([1.0])
        return soft_labels, feat, confidence

def meld_feats():
    '''
    trn and test: total 12599 and empty 142 and no actspk 4031
    val: total 1109 and empty 12 and no actspk 354
    '''
    for setname in ['train', 'val', 'test']:
        empty_video = 0
        noactspk_video = 0
        total_video = 0
        target = json.load(open(target_path.format(setname)))
        data = h5py.File(h5_fts_path.format(setname))
        segment_ids = list(data[setname].keys())
        print('{} total {} segments'.format(setname, len(segment_ids)))
        for i, segment_id in enumerate(segment_ids):
            # val-dia0_utt0.npz
            outputfilename = setname + '-' + segment_id + ".npz"
            # new setmentid = val/dia0_utt0
            segment_id = f'{setname}/{segment_id}'
            outputfile = os.path.join(npyz_dir, outputfilename)
            total_video += 1
            if os.path.exists(outputfile):
                continue
            # print(segment_id)
            if len(data[segment_id]) == 0:
                empty_video += 1
                soft_labels, feat, confidence = empty_data_process(target, segment_id, do_raw_img)   
            else:
                # 挑选特征的策略
                soft_labels = np.array(data[segment_id]['pred'])
                confidence = np.array(data[segment_id]['confidence'])
                has_active_spk = np.array(data[segment_id]['has_active_spk'])
                feat = np.array(data[segment_id][ft_key])
                # print(data[segment_id])
                if not has_active_spk:
                    # 如果有 active_spk 那么特征是真实的特征序列, 如果没有则进行下面的处理
                    # print('[Debug] has_active_spk {}'.format(has_active_spk))
                    noactspk_video += 1
                    # 如果没有 active_spk 那么执行策略
                    frames_idxs = np.array(data[segment_id]['frames_idx'])
                    spk2idx = {}
                    for i, idx in enumerate(frames_idxs):
                        # 只考虑大于0.1的脸，尽量最后能留一些好的脸
                        if confidence[i] > 0.1:
                            # 由于spk=0的时候 1 = int(0001)
                            if idx < 1000:
                                spk = 0
                            else:
                                spk = int(str(idx)[0])
                            if spk2idx.get(spk):
                                spk2idx[spk] += [i]
                            else:
                                spk2idx[spk] = [i]
                    if len(spk2idx) == 0:
                        # 所有的人脸的阈值都小于 0.1
                        soft_labels, feat, confidence = empty_data_process(target, segment_id, do_raw_img)
                    else:
                        max_spk , max_spk_frames = 0, 0
                        for spk in spk2idx.keys():
                            if len(spk2idx[spk]) > max_spk_frames:
                                max_spk = spk
                                max_spk_frames = len(spk2idx[spk])
                        # print(f'[Debug] {segment_id} selected spk is: {max_spk}')
                        indexs = spk2idx[max_spk]
                        confidence = confidence[indexs]
                        soft_labels = soft_labels[indexs]
                        feat = feat[indexs]
                        # print(indexs)
                        # print(frames_idxs)
                        # print(spk2idx)
                        # print('len {} ft {}'.format(len(indexs), feat.shape))
            if not do_raw_img:
                assert len(soft_labels) == len(feat)
                np.savez_compressed(outputfile,
                                    confidence=confidence.astype(np.float16),
                                    soft_labels=soft_labels.astype(np.float16),
                                    features=feat.astype(np.float16))
            else:
                # 将图片进行归一化并转化为灰度图
                imgs = []
                for img in feat:
                    # print(img.shape, np.sum(img))
                    img = prepocess_img_data(img, mean, std, img_size)
                    imgs.append(img)
                imgs = np.array(imgs)
                np.savez_compressed(outputfile,
                                    confidence=confidence.astype(np.float16),
                                    features=imgs.astype(np.float16))
                # print('{} feat {}'.format(setname, imgs.shape))
        print('total {} and empty {} and no actspk {}'.format(total_video, empty_video, noactspk_video))

if __name__ == '__main__':
    meld_feats()