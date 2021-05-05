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
from preprocess.tasks.audio import ComParEExtractor, Wav2VecExtractor
import cv2

'''
处理的meld数据，如果一段视频里面有active-spk，那么保存当前spk的 frame indexs，
这时保存的正常的index，并不是2002这种格式。 如果一段视频里面没有 active-spk，
那么保存所有人的人脸信息，但是呢，如果spk=0，那么这时候也不是 0001 这种形式(int(0001)=1)，所以同时存在 1 2 3 1001 1002 2001 两种形式
'''

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

def extract_one_video_mid_layers(video_dir, denseface_model, face_selector, detect_type):
    '''
    存储的时候frame-idx, 如果有 active_spk_id 那么存储形式是该 spk 的frames, 比如 1 2 3 等
    如果没有 active_spk_id 那么存储形式是该 所有spk的frames, 比如 1001 2001 3001 等
    '''
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

def get_face_dir_openface(utt_id):
    # Ses01F_impro06_F002
    root = '/data7/emobert/exp/evaluation/MELD/faces/'
    return os.path.join(root, utt_id)

def split_h5(all_h5_path, save_root='feature/denseface'):
    target_root = '/data7/emobert/exp/evaluation/MELD/target'
    trn_int2name = np.load(os.path.join(target_root, 'train', f'int2name.npy'))
    trn_int2name = [transform_utt_id(utt_id, 'train') for utt_id in trn_int2name[:, 0].tolist()]
    val_int2name = np.load(os.path.join(target_root, 'val', f'int2name.npy'))
    val_int2name = [transform_utt_id(utt_id, 'val') for utt_id in val_int2name[:, 0].tolist()]
    tst_int2name = np.load(os.path.join(target_root, 'test', f'int2name.npy'))
    tst_int2name = [transform_utt_id(utt_id, 'test') for utt_id in tst_int2name[:, 0].tolist()]
    print('trn {} val {} and test {}'.format(len(trn_int2name), len(val_int2name), len(tst_int2name)))
    h5f = h5py.File(all_h5_path, 'r')
    split_by_utt_id(h5f, val_int2name, os.path.join(save_root, 'val.h5'))
    split_by_utt_id(h5f, tst_int2name, os.path.join(save_root, 'test.h5'))
    split_by_utt_id(h5f, trn_int2name, os.path.join(save_root, 'train.h5'))
    
    h5f.close()

def split_by_utt_id(in_h5f, utt_ids, save_path):
    out_h5f = h5py.File(save_path, 'w')
    for utt_id in tqdm(utt_ids):
        if utt_id not in in_h5f.keys():
            out_h5f[utt_id] = np.zeros(0)
            print('[Warning] utt {} is empty'.format(utt_id))
            continue
        tgt = in_h5f[utt_id]
        # print('{} In all: {}'.format(utt_id, tgt['frames_idx'][0]))
        if isinstance(tgt, h5py._hl.dataset.Dataset):
            out_h5f[utt_id] = deepcopy(tgt[()])
        elif isinstance(tgt, h5py._hl.group.Group):
            _group = out_h5f.create_group(utt_id)
            for key in tgt.keys():
                _group[key] = deepcopy(tgt[key][()])
        # print('{} In set: {}'.format(utt_id, out_h5f[utt_id]))
    out_h5f.close()

def transform_utt_id(utt_id, set_name):
    dia_num, utt_num = utt_id.split('_')
    return f'{set_name}/dia{dia_num}_utt{utt_num}'

def get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/MELD/target'):
    ans = []
    for set_name in ['train', 'val', 'test']:
        utt_ids = np.load(os.path.join(target_root, set_name, f'int2name.npy'))
        utt_ids = [transform_utt_id(utt_id, set_name) for utt_id in utt_ids[:, 0].tolist()]
        ans += utt_ids
    return ans

def extract_comparE_file(audio_path, extractor_model):
    # for one audio clip, audio_path = audio_dir + {set_name}/dia{dia_num}_utt{utt_num}
    audio_path = audio_path + '.wav'
    # print(audio_path)
    if not os.path.exists(audio_path):
        print(f'[Not exist] {audio_path}')
        feat = np.zeros([1,130])
        frame_nums =np.array(1)
    else:
        feat = extractor_model(audio_path)
        frame_nums = np.array(len(feat))
    return {'feat': feat, 'frame_idx': frame_nums}

def extract_wav2vec_file(audio_path, extractor_model):
    # for one audio clip, audio_path = audio_dir + {set_name}/dia{dia_num}_utt{utt_num}
    audio_path = audio_path + '.wav'
    if not os.path.exists(audio_path):
        print(f'[Not exist] {audio_path}')
        feat = np.zeros([1, 768])
        frame_nums =np.array(1)
    else:
        feat = extractor_model(audio_path)
        frame_nums = np.array(len(feat))
        if len(feat.shape) == 3:
            # batchsize=1
            feat = feat[0]
    return {'feat': feat, 'frame_idx': frame_nums}

def transh5_format(save_path, new_save_path):
    data = h5py.File(save_path)
    out_h5f = h5py.File(new_save_path, 'w')
    for setname in data.keys():
        for uttid in data[setname].keys():
            new_uttid = setname + '-' + uttid
            tgt = data[setname][uttid]['feat']
            if isinstance(tgt, h5py._hl.dataset.Dataset):
                out_h5f[new_uttid] = deepcopy(tgt[()])
            elif isinstance(tgt, h5py._hl.group.Group):
                _group = out_h5f.create_group(new_uttid)
                for key in tgt.keys():
                    _group[key] = deepcopy(tgt[key][()])
    out_h5f.close()

if __name__ == '__main__':
    
    output_dir = '/data7/emobert/exp/evaluation/MELD/feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if False:
        detect_type = sys.argv[1]
        if detect_type == 'seetaface':
            name = "denseface_seetaface_meld_mean_std_torch"
        elif detect_type == 'openface':
            name = "denseface_openface_meld_mean_std_torch"
        else:
            raise ValueError('detect type must be openface or seetaface')

        if not os.path.exists(output_dir + '/' + name):
            os.mkdir(output_dir + '/' + name)
        # for visual face
        utt_ids = get_all_utt_ids()
        images_mean, images_std = 67.61417, 37.89171
        save_path = os.path.join(output_dir, name, 'all.h5')
        denseface = DensefaceExtractor(mean=images_mean, std=images_std)
        denseface.register_midlayer_hook([
            "features.transition1.relu",
            "features.transition2.relu"
        ])
        face_selector = FaceSelector()
        extract_func = partial(extract_one_video_mid_layers, denseface_model=denseface, face_selector=face_selector, detect_type=detect_type)
        if detect_type == 'seetaface':
            raise NotImplemented()
            extract_features_h5(extract_func, get_face_dir_seetaface, utt_ids, save_path)
        else:
            extract_features_h5(extract_func, get_face_dir_openface, utt_ids, save_path)
        split_h5(save_path, save_root=os.path.join(output_dir, name))
    
     # # 偏函数: 主要目的是冻结固定参数？将所作用的函数作为 partial() 函数的第一个参数，原函数的各个参数依次作为 partial（）函数的后续参数，
    # # 原函数有关键字参数的一定要带上关键字，没有的话，按原有参数顺序进行补充.

    if False:
        # for speech comparE
        audio_feature_dir = os.path.join(output_dir, 'comparE_raw')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        comparE_model = ComParEExtractor(tmp_dir=f'{output_dir}/raw_fts')
        extract_comparE = partial(extract_comparE_file, extractor_model=comparE_model)
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        audio_dir = '/data7/MEmoBert/emobert/exp/evaluation/MELD/audio'
        utt_ids = get_all_utt_ids()
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
        # trans h5 format, val-dia0_utt0.npz
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        new_save_path = os.path.join(audio_feature_dir, 'all.h5')
        transh5_format(save_path, new_save_path)

    if True:
        # for speech wav2vec2.0 
        use_asr_based_model = False
        if use_asr_based_model:
            audio_feature_dir = os.path.join(output_dir, 'wav2vec_raw_asr')
        else:
            audio_feature_dir = os.path.join(output_dir, 'wav2vec_raw')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        wav2vec_model = Wav2VecExtractor(downsample=-1, gpu=0, use_asr_based_model=use_asr_based_model)
        extract_wav2vec = partial(extract_wav2vec_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        audio_dir = '/data7/MEmoBert/emobert/exp/evaluation/MELD/audio'
        utt_ids = get_all_utt_ids()
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
        # trans h5 format, val-dia0_utt0.npz
        save_path = os.path.join(audio_feature_dir, 'all_set.h5')
        new_save_path = os.path.join(audio_feature_dir, 'all.h5')
        transh5_format(save_path, new_save_path)

# PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=0 python extract_denseface.py openface
# PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=0 python extract_denseface.py seetaface