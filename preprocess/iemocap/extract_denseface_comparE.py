import os, glob
import numpy as np
import h5py
from copy import deepcopy
import sys
sys.path.append('/data7/MEmoBert/preprocess/tasks')
from functools import partial
from tqdm import tqdm
from toolz.sandbox import unzip
import torch.nn.functional as F
from preprocess.tasks.vision import DensefaceExtractor, FaceSelector
from preprocess.extract_features import extract_denseface_trans_dir

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
            continue
    h5f.close()

def extract_one_video(video_dir, denseface_model, detect_type):
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

def extract_one_video_mid_layers(video_dir, denseface_model, detect_type):
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
    feats, preds, trans1s, trans2s, raw_imgs = [], [], [], []
    for img in imgs:
        feat, pred = denseface_model(img)
        feats.append(feat)
        preds.append(pred)
        trans1, trans2 = denseface_model.get_mid_layer_output()
        trans1 = F.avg_pool2d(trans1, kernel_size=32, stride=1).view(trans1.size(0), -1).detach().cpu().numpy()
        trans2 = F.avg_pool2d(trans2, kernel_size=16, stride=1).view(trans2.size(0), -1).detach().cpu().numpy()
        trans1s.append(trans1)
        trans2s.append(trans2)
        raw_imgs.append(img)
    feats = np.concatenate(feats, axis=0)
    preds = np.concatenate(preds, axis=0)
    trans1s = np.concatenate(trans1s, axis=0)
    trans2s = np.concatenate(trans2s, axis=0)
    return {'feat': feats, 'pred': preds, 'trans1': trans1s, 'trans2': trans2s}

def get_face_dir_seetaface(utt_id):
    # Ses01F_impro06_F002
    session_id = utt_id[4]
    return '/data7/emobert/exp/evaluation/IEMOCAP/Session{}/face/{}'.format(session_id, utt_id)

def get_face_dir_openface(utt_id):
    # Ses01F_impro06_F002
    session_id = utt_id[4]
    return '/data7/emobert/exp/evaluation/IEMOCAP/Session{}/openface/{}/{}_aligned'.format(session_id, utt_id, utt_id)

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
    for cv in range(1, 11):
        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trn_int2name, val_int2name, tst_int2name, _, _, _ = get_trn_val_tst(cv, target_root='/data7/emobert/exp/evaluation/IEMOCAP/target')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name.tolist()))
        val_int2name = list(map(lambda x: x[0].decode(), val_int2name.tolist()))
        tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name.tolist()))
        split_by_utt_id(h5f, trn_int2name, os.path.join(save_dir, 'trn.h5'))
        split_by_utt_id(h5f, val_int2name, os.path.join(save_dir, 'val.h5'))
        split_by_utt_id(h5f, tst_int2name, os.path.join(save_dir, 'tst.h5'))
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

def get_all_utt_ids():
    target_root = '/data7/emobert/exp/evaluation/IEMOCAP/target/1'
    ans = []
    for set_name in ['trn', 'val', 'tst']:
        utt_ids = np.load(os.path.join(target_root, f'{set_name}_int2name.npy')).tolist()
        utt_ids = list(map(lambda x: x[0].decode('utf8'), utt_ids))
        ans += utt_ids
    return ans
    
if __name__ == '__main__':
    
    # for face
    if False:
        detect_type = sys.argv[1]
        output_dir = '/data7/MEmoBert/evaluation/IEMOCAP/feature'
        if detect_type == 'seetaface':
            name = "denseface_seetaface_iemocap_mean_std_torch"
        elif detect_type == 'openface':
            name = "denseface_openface_iemocap_mean_std_torch"
        else:
            raise ValueError('detect type must be openface or seetaface')

        if not os.path.exists(output_dir + '/' + name):
            os.mkdir(output_dir + '/' + name)
        utt_ids = get_all_utt_ids()
        images_mean = 131.0754
        images_std = 47.8581
        denseface = DensefaceExtractor(mean=images_mean, std=images_std)
        denseface.register_midlayer_hook([
            "features.transition1.relu",
            "features.transition2.relu"
        ])
        extract_func = partial(extract_one_video_mid_layers, denseface_model=denseface, detect_type=detect_type)
        save_path = os.path.join(output_dir, name, 'all.h5')
        if detect_type == 'seetaface':
            extract_features_h5(extract_func, get_face_dir_seetaface, utt_ids, save_path)
        else:
            extract_features_h5(extract_func, get_face_dir_openface, utt_ids, save_path)
        save_path = os.path.join(output_dir, name, 'all.h5')
        split_h5(save_path, save_root=os.path.join(output_dir, name))

    if True:
        # for speech
        print('Start to split')
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature/comparE_raw'
        save_path = os.path.join(output_dir, 'all.h5')
        split_h5(save_path, save_root=output_dir)
    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=7 python extract_denseface.py openface/seetaface