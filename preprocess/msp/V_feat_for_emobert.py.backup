import os, sys, glob
import cv2
import h5py
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
from copy import deepcopy
import traceback

sys.path.append('/data7/MEmoBert/preprocess/tasks')
from functools import partial
from tqdm import tqdm
from toolz.sandbox import unzip
from vision import DensefaceExtractor

def extract_features_h5(extract_func, get_input_func, utt_ids, save_path):
    if os.path.exists(save_path):
        try:
            _h5f = h5py.File(save_path, 'r')
            if len(_h5f.keys()) == len(utt_ids):
                _h5f.close()
                return
            _h5f.close()
        except OSError:
            os.remove(save_path)

    h5f = h5py.File(save_path, 'w')

    for utt_id in tqdm(utt_ids):
        input_param = get_input_func(utt_id)
        feature = extract_func(input_param)
        if isinstance(feature, dict):
            utt_data = h5f.create_group(utt_id)
            for k, v in feature.items():
                utt_data[k] = v
        elif isinstance(feature, np.ndarray):
            h5f[utt_id] = feature
        else:
            print('Uttid:{} has some problem, double check it'.format(utt_id))
            exit(1)
    h5f.close()


def extract_one_video(video_dir, denseface_model):
    if detect_type == 'seetaface':
        imgs = glob.glob(video_dir+'/*.jpg')
    else:
        imgs = glob.glob(video_dir+'/*.bmp')
    if len(imgs) == 0:
        print(video_dir, 'has no imgs, double check it')
        return None
    feat_pred = [denseface_model(x) for x in imgs]
    feats, pred = map(list, unzip(feat_pred))
    feats = np.concatenate(feats, axis=0)
    pred = np.concatenate(pred, axis=0)
    return {'feat': feats, 'pred': pred}

def get_face_dir_seetaface(utt_id):
    return '/data7/emobert/exp/evaluation/MSP-IMPROV/Face/{}'.format(utt_id)

def get_face_dir_openface(utt_id):
    return '/data7/emobert/exp/evaluation/MSP-IMPROV/OpenFace/{}/{}_aligned'.format(utt_id, utt_id)

def get_trn_val_tst(cv, target_root='target'):
    target_dir = os.path.join(target_root, str(cv))
    trn_int2name = np.load(os.path.join(target_dir, 'trn_int2name.npy'))
    val_int2name = np.load(os.path.join(target_dir, 'val_int2name.npy'))
    tst_int2name = np.load(os.path.join(target_dir, 'tst_int2name.npy'))
    trn_label = np.load(os.path.join(target_dir, 'trn_label.npy'))
    val_label = np.load(os.path.join(target_dir, 'val_label.npy'))
    tst_label = np.load(os.path.join(target_dir, 'tst_label.npy'))
    assert len(trn_int2name) == len(trn_label)
    assert len(val_int2name) == len(val_label)
    assert len(tst_int2name) == len(tst_label)
    return trn_int2name, val_int2name, tst_int2name, trn_label, val_label, tst_label

def split_h5(all_h5, save_root='feature/denseface'):
    h5f = h5py.File(all_h5, 'r')
    for cv in range(1, 13):
        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        trn_int2name, val_int2name, tst_int2name, _, _, _ = get_trn_val_tst(cv, target_root='/data6/lrc/MSP-IMPROV_feature/target/cv_level')
        trn_int2name = trn_int2name.tolist()
        val_int2name = val_int2name.tolist()
        tst_int2name = tst_int2name.tolist()

        # for 11:1 setting
        trn_int2name += tst_int2name
        tst_int2name = val_int2name

        split_by_utt_id(h5f, trn_int2name, os.path.join(save_dir, 'trn.h5'))
        split_by_utt_id(h5f, val_int2name, os.path.join(save_dir, 'val.h5'))
        split_by_utt_id(h5f, tst_int2name, os.path.join(save_dir, 'tst.h5'))
    
    h5f.close()

def split_by_utt_id(in_h5f, utt_ids, save_path):
    out_h5f = h5py.File(save_path, 'w')
    for utt_id in tqdm(utt_ids):
        # print(utt_id)
        if utt_id not in in_h5f.keys():
            out_h5f[utt_id] = np.zeros(0)
            continue
        tgt = in_h5f[utt_id]
        if isinstance(tgt, h5py._hl.dataset.Dataset):
            out_h5f[utt_id] = deepcopy(tgt[()])
        elif isinstance(tgt, h5py._hl.group.Group):
            _group = out_h5f.create_group(utt_id)
            for key in tgt.keys():
                _group[key] = deepcopy(tgt[key][()])
    out_h5f.close()
    
def get_utt_ids():
    utt_file = '/data6/lrc/MSP-IMPROV_feature/target/all_int2name.npy'
    return np.load(utt_file)

if __name__ == '__main__':
    detect_type = sys.argv[1]
    output_dir = '/data7/emobert/exp/evaluation/MSP-IMPROV/feature'
    if detect_type == 'seetaface':
        name = "denseface_seetaface_mean_std_MSP"
    elif detect_type == 'openface':
        name = "denseface_openface_mean_std_MSP"
    else:
        raise ValueError('detect type must be openface or seetaface')

    if not os.path.exists(output_dir + '/' + name):
        os.mkdir(output_dir + '/' + name)

    utt_ids = get_utt_ids()
    # movie without mask
    # images_mean=63.987095
    # images_std=43.00519
    # MSP image mean & std
    if detect_type == 'openface':
        images_mean = 113.30168
        images_std = 42.49455
    elif detect_type == 'seetaface':
        images_mean = 113.59194
        images_std = 45.201733
    else:
        print('detect_type is Error! {}'.format(detect_type))
        images_mean, images_std = None, None
    
    restore_path = '/data2/zjm/tools/FER_models/denseface/DenseNet-BC_growth-rate12_depth100_FERPlus/model/epoch-200'
    model = DensefaceExtractor(restore_path, mean=images_mean, std=images_std)
    extract_func = partial(extract_one_video, denseface_model=model)
    save_path = os.path.join(output_dir, name, 'all.h5')
    if detect_type == 'seetaface':
        extract_features_h5(extract_func, get_face_dir_seetaface, utt_ids, save_path)
    else:
        extract_features_h5(extract_func, get_face_dir_openface, utt_ids, save_path)
    save_path = os.path.join(output_dir, name, 'all.h5')
    split_h5(save_path, save_root=os.path.join(output_dir, name))


    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=5 python denseface_new.py seetaface
    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=5 python denseface_new.py openface