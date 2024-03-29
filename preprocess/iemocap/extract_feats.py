import os, glob
import shutil
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
from preprocess.tasks.audio import ComParEExtractor, Wav2VecExtractor, RawWavExtractor, Wav2VecCNNExtractor
from preprocess.MELD.extract_denseface_comparE import extract_comparE_file, extract_features_h5, extract_wav2vec_file, extract_rawwav_file

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
    feats, preds, trans1s, trans2s, raw_imgs = [], [], [], [], []
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

def get_all_utt_ids(target_root):
    ans = []
    for set_name in ['trn', 'val', 'tst']:
        utt_ids = np.load(os.path.join(target_root, f'{set_name}_int2name.npy')).tolist()
        utt_ids = list(map(lambda x: x[0].decode('utf8'), utt_ids))
        ans += utt_ids
    return ans

def get_3mean_speech_data(save_path, save_3mean_path):
    h5f = h5py.File(save_path, 'r')
    avg_lens = []
    pooling_num_frames = 3
    out_h5f = h5py.File(save_3mean_path, 'w')
    for uttId in h5f.keys():
        feat = h5f[uttId]['feat']
        mean_norm_feat = []
        for i in range(0, len(feat), pooling_num_frames):
            if i+pooling_num_frames >= len(feat):
                mean_norm_feat.append(np.mean(feat[i:], axis=0))
            else:
                mean_norm_feat.append(np.mean(feat[i:i+pooling_num_frames], axis=0))
        if len(mean_norm_feat) == 0:
            print('[Afer Mean]segment {} meam-norm {}'.format(segment_index, len(mean_norm_feat)))
        feat = np.array(mean_norm_feat)
        avg_lens.append(len(feat))
        # save to h5
        out_h5f[uttId] = feat
    print('samples {} and avg len {}'.format(len(avg_lens), sum(avg_lens)/len(avg_lens)))
    out_h5f.close()

if __name__ == '__main__':
    
    # for face
    if False:
        detect_type = sys.argv[1]
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature'
        if detect_type == 'seetaface':
            name = "denseface_seetaface_iemocap_mean_std_torch"
            model_path = None
        elif detect_type == 'openface':
            name = "denseface_openface_iemocap_mean_std_torch"
            model_path = None
        elif detect_type == 'openface_affectnet':
            name = "denseface_affectnet_openface_iemocap_mean_std_torch"
            model_path = '/data9/datasets/AffectNetDataset/combine_with_fer/results/densenet100_adam0.0002_0.0/ckpts/model_step_12.pt'
        else:
            raise ValueError('detect type must be openface or seetaface')

        if not os.path.exists(output_dir + '/' + name):
            os.mkdir(output_dir + '/' + name)
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/IEMOCAP/target/1')
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

    if False:
        # for speech
        print('Start to split')
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature/comparE_raw'
        save_path = os.path.join(output_dir, 'all.h5')
        split_h5(save_path, save_root=output_dir)
    

    if False:
        wavscp_path = '/data2/zjm/speech_emotion/IEMOCAP/data/wav.scp'
        audio_dir = '/data7/emobert/exp/evaluation/IEMOCAP/audio/'
        with open(wavscp_path) as f:
            lines = f.readlines()
        for line in lines:
            path = line.strip('\n').split(' ')[1]
            shutil.copy(path, audio_dir)

    if False:
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
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
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/IEMOCAP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/IEMOCAP/target/1')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)
    
    if False:
        # for speech wav2vec
        print('Start to split')
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_raw'
        save_path = os.path.join(output_dir, 'all.h5')
        split_h5(save_path, save_root=output_dir)
    
    if True:
        # for speech wav2vec
        print('Start to split')
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_raw'
        save_path = os.path.join(output_dir, 'all.h5')
        output_3mean_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature/wav2vec_raw_3mean'
        save_3mean_path = os.path.join(output_3mean_dir, 'all.h5')
        get_3mean_speech_data(save_path, save_3mean_path)
        split_h5(save_3mean_path, save_root=output_3mean_dir)

    if False:
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # for speech 
        audio_feature_dir = os.path.join(output_dir, 'wav2vec_rawwav')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        model_path = '/data7/emobert/resources/pretrained/wav2vec_base'
        wav2vec_model = RawWavExtractor(model_path, max_seconds=8)
        extract_rawwav = partial(extract_rawwav_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/IEMOCAP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/IEMOCAP/target/1')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_rawwav, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)


    if False:
        output_dir = '/data7/emobert/exp/evaluation/IEMOCAP/feature'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # for speech wav2vec2.0
        audio_feature_dir = os.path.join(output_dir, 'wav2vec_cnn')
        if not os.path.exists(audio_feature_dir):
            os.mkdir(audio_feature_dir)
        wav2vec_model = Wav2VecCNNExtractor(gpu=0)
        extract_wav2vec = partial(extract_wav2vec_file, extractor_model=wav2vec_model)
        save_path = os.path.join(audio_feature_dir, 'all.h5')
        audio_dir = '/data7/emobert/exp/evaluation/IEMOCAP/audio'
        utt_ids = get_all_utt_ids(target_root='/data7/emobert/exp/evaluation/IEMOCAP/target/1')
        print('total {} uttids'.format(len(utt_ids)))
        extract_features_h5(extract_wav2vec, lambda x: os.path.join(audio_dir, x),  utt_ids, save_path)

    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=0 python extract_feats.py openface_affectnet/openface/seetaface/
    # PYTHONPATH=/data7/MEmoBert CUDA_VISIBLE_DEVICES=4 python extract_feats.py