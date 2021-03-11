import os, glob
import numpy as np
import h5py
from copy import deepcopy
import sys
sys.path.append('/data7/MEmoBert/preprocess/tasks')
import pandas as pd
from functools import partial
from tqdm import tqdm
from toolz.sandbox import unzip
import torch.nn.functional as F
from preprocess.tasks.vision import DensefaceExtractor, FaceSelector
from preprocess.extract_features import extract_denseface_trans_dir

import cv2

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
        if h5f.get(utt_id):
            continue
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
            continue
    h5f.close()

def extract_one_video(video_dir, denseface_model):
    using_frame = False
    if detect_type == 'seetaface':
        imgs = glob.glob(video_dir+'/*.jpg')
        imgs = sorted(imgs, key=lambda x:int(x.split('/')[-1][:-4]))
        frames = glob.glob(video_dir.replace('face', 'frame')+'/*.jpg')
    else:
        imgs = glob.glob(video_dir+'/*.bmp')
        imgs = sorted(imgs, key=lambda x:int(x.split('/')[-1].split('_')[-1][:-4]))
        frames = glob.glob('/'.join(video_dir.replace('openface', 'frame').split('/')[:-1])+'/*.jpg')
    frames = sorted(frames, key=lambda x:int(x.split('/')[-1][:-4]))
    if len(imgs) == 0:
        print(video_dir, 'has no imgs, double check it')
        if using_frame:
            imgs = frames
            print(f'Using frames instead. frame len:{len(frames)}')
            if len(frames) == 0:
                return None
        else:
            return None
    feat_pred = [denseface_model(x) for x in imgs]
    feats, pred = map(list, unzip(feat_pred))
    feats = np.concatenate(feats, axis=0)
    pred = np.concatenate(pred, axis=0)
    return {'feat': feats, 'pred': pred}

def get_sort_index(path):
    name = os.path.basename(path).split('.')[0]
    spk_id, utt_id = name.split('_')[-2:]
    return int(spk_id) * 1000 + int(utt_id)

def get_confidence(imgs, conf_df):
    ret = []
    for img in imgs:
        name = os.path.basename(img).split('.')[0]
        spk_id, frame_id = name.split('_')[-2:]
        face_ridx = conf_df[' face_id'] == int(spk_id)
        frame_ridx = conf_df['frame'] == int(frame_id)
        conf = conf_df[face_ridx & frame_ridx][' confidence']
        assert len(conf) == 1, conf
        ret.append(conf.iloc[0])
    return ret

def extract_one_video_mid_layers(video_dir, denseface_model, face_selector):
    using_frame = False
    if detect_type == 'seetaface':
        raise NotImplementedError()
        # imgs = glob.glob(video_dir+'/*.jpg')
        # imgs = sorted(imgs, key=lambda x:int(x.split('/')[-1][:-4]))
        # frames = glob.glob(video_dir.replace('face', 'frame')+'/*.jpg')
    else:
        active_spk_id = open(os.path.join(video_dir, 'has_active_spk.txt')).read().strip()
        if active_spk_id == "None":
            active_spk_flag = False
            utt_id = video_dir.rstrip('/').split('/')[-1]
            imgs = glob.glob(os.path.join(video_dir, f'{utt_id}_aligned/*.bmp'))
            if len(imgs) == 0:
                return None
            imgs = sorted(imgs, key=lambda x: get_sort_index(x))
            frames_idx = [get_sort_index(x) for x in imgs]
            data_frame = pd.read_csv(os.path.join(video_dir, os.path.basename(video_dir).rstrip('/')+'.csv'))
            confidence = get_confidence(imgs, data_frame)
            # return None
        else:
            active_spk_flag = True
            active_spk_id = int(active_spk_id)
            ret_slc = face_selector(video_dir, active_spk_id)
            imgs = [x['img'] for x in ret_slc]
            frames_idx = [get_sort_index(x) for x in imgs]
            confidence = [x['confidence'] for x in ret_slc]
            imgs = sorted(imgs, key=lambda x:int(x.split('/')[-1].split('_')[-1][:-4]))
        if using_frame:
            frames = glob.glob(video_dir.replace('faces', 'frame')+'/*.jpg')
            frames = sorted(frames, key=lambda x:int(x.split('/')[-1][:-4]))
        
    if len(imgs) == 0:
        print(video_dir, 'has no imgs, double check it')
        if using_frame:
            imgs = frames
            print(f'Using frames instead. frame len:{len(frames)}')
            if len(frames) == 0:
                return None
        else:
            return None
    feats, preds, trans1s, trans2s, img_datas = [], [], [], [], []
    for img in imgs:
        img_data = cv2.imread(img)
        img_datas.append(img_data)
        feat, pred = denseface_model(img)
        feats.append(feat)
        preds.append(pred)
        trans1, trans2 = denseface_model.get_mid_layer_output()
        trans1 = F.avg_pool2d(trans1, kernel_size=32, stride=1).view(trans1.size(0), -1).detach().cpu().numpy()
        trans2 = F.avg_pool2d(trans2, kernel_size=16, stride=1).view(trans2.size(0), -1).detach().cpu().numpy()
        trans1s.append(trans1)
        trans2s.append(trans2)
    feats = np.concatenate(feats, axis=0)
    preds = np.concatenate(preds, axis=0)
    trans1s = np.concatenate(trans1s, axis=0)
    trans2s = np.concatenate(trans2s, axis=0)
    img_datas = np.array(img_datas)
    # print(feats.shape)
    # print(img_datas.shape)
    # input()
    return {'feat': feats, 'pred': preds, 'trans1': trans1s, 'trans2': trans2s, 'confidence': confidence, 'frames_idx': frames_idx, 'has_active_spk': active_spk_flag, 'img_data':img_datas}

# def get_face_dir_seetaface(utt_id):
#     # Ses01F_impro06_F002
#     session_id = utt_id[4]
#     return '/data7/MEmoBert/evaluation/IEMOCAP/Session{}/face/{}'.format(session_id, utt_id)

def get_face_dir_openface(utt_id):
    # Ses01F_impro06_F002
    root = '/data7/MEmoBert/evaluation/MELD/faces'
    return os.path.join(root, utt_id)

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
    target_root = '/data7/MEmoBert/evaluation/MELD/target'
    trn_int2name = np.load(os.path.join(target_root, 'train', f'int2name.npy'))
    trn_int2name = [transform_utt_id(utt_id, 'train') for utt_id in trn_int2name[:, 0].tolist()]
    val_int2name = np.load(os.path.join(target_root, 'val', f'int2name.npy'))
    val_int2name = [transform_utt_id(utt_id, 'train') for utt_id in val_int2name[:, 0].tolist()]
    tst_int2name = np.load(os.path.join(target_root, 'test', f'int2name.npy'))
    tst_int2name = [transform_utt_id(utt_id, 'train') for utt_id in tst_int2name[:, 0].tolist()]

    split_by_utt_id(h5f, trn_int2name, os.path.join(save_root, 'trn.h5'))
    split_by_utt_id(h5f, val_int2name, os.path.join(save_root, 'val.h5'))
    split_by_utt_id(h5f, tst_int2name, os.path.join(save_root, 'tst.h5'))
    
    h5f.close()

def split_by_utt_id(in_h5f, utt_ids, save_path):
    out_h5f = h5py.File(save_path, 'w')
    for utt_id in tqdm(utt_ids):
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

def transform_utt_id(utt_id, set_name):
    dia_num, utt_num = utt_id.split('_')
    return f'{set_name}/dia{dia_num}_utt{utt_num}'

def get_all_utt_ids():
    target_root = '/data7/MEmoBert/evaluation/MELD/target'
    ans = []
    for set_name in ['train', 'val', 'test']:
        utt_ids = np.load(os.path.join(target_root, set_name, f'int2name.npy'))
        utt_ids = [transform_utt_id(utt_id, set_name) for utt_id in utt_ids[:, 0].tolist()]
        ans += utt_ids
    return ans
    

if __name__ == '__main__':
    detect_type = sys.argv[1]
    output_dir = '/data7/MEmoBert/evaluation/MELD/feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if detect_type == 'seetaface':
        name = "denseface_seetaface_meld_mean_std_torch"
    elif detect_type == 'openface':
        name = "denseface_openface_meld_mean_std_torch"
    else:
        raise ValueError('detect type must be openface or seetaface')

    if not os.path.exists(output_dir + '/' + name):
        os.mkdir(output_dir + '/' + name)

    utt_ids = get_all_utt_ids()
    # msp
    images_mean = 67.61417
    images_std = 37.89171

    denseface = DensefaceExtractor(mean=images_mean, std=images_std)
    denseface.register_midlayer_hook([
        "features.transition1.relu",
        "features.transition2.relu"
    ])
    face_selector = FaceSelector()
    extract_func = partial(extract_one_video_mid_layers, denseface_model=denseface, face_selector=face_selector)
    save_path = os.path.join(output_dir, name, 'all.h5')
    if detect_type == 'seetaface':
        raise NotImplemented()
        extract_features_h5(extract_func, get_face_dir_seetaface, utt_ids, save_path)
    else:
        extract_features_h5(extract_func, get_face_dir_openface, utt_ids, save_path)
    save_path = os.path.join(output_dir, name, 'all.h5')
    split_h5(save_path, save_root=os.path.join(output_dir, name))
    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=0 python extract_denseface.py openface
    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=0 python extract_denseface.py seetaface