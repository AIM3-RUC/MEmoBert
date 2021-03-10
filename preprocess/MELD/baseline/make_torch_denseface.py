import os
import glob
import torch
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import sys
from preprocess.denseface.model.dense_net import DenseNet

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(os.path.join(target_root_dir, str(cv), '{}_int2name.npy'.format(setname)))
    int2label = np.load(os.path.join(target_root_dir, str(cv), '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def get_faces(face_dir):
    face_lst = glob.glob(os.path.join(face_dir, '*.jpg'))
    face_lst = sorted(face_lst, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return face_lst

def get_face_tensors(face_lst):
    img_size = 64
    images_mean = 131.0754
    images_std = 47.858177
    imgs = []
    for face in face_lst:
        img = cv2.imread(face)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        imgs.append(img)
    imgs = (np.array(imgs, np.float32) - images_mean) / images_std
    imgs = np.expand_dims(imgs, 3)
    return imgs

# def chunk(iterable, chunk_size):
#     ret = []
#     idx = 0
#     while idx < len(iterable):
#         ret.append(iterable[idx])
#         idx += 1
#         if len(ret) == chunk_size:
#             yield ret
#             ret = []
#     yield ret    

def chunk(lst, chunk_size):
    idx = 0
    while chunk_size * idx < len(lst):
        yield lst[idx*chunk_size: (idx+1)*chunk_size]
        idx += 1

def make_all_new_denseface(config):
    device = torch.device('cuda:0')
    extractor = DenseNet(0, **model_cfg)
    model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
    extractor.load_state_dict(torch.load(model_path))
    extractor.eval()
    extractor.to(device)
    trn_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'trn')
    val_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_root'], 1, 'tst')
    trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
    val_int2name = list(map(lambda x: x[0].decode(), val_int2name))
    tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name))
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['feature_root'], 'denseface_torch', 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        ses_id = utt_id[4]
        face_dir = os.path.join(config['face_root'], f'Session{ses_id}', 'face', utt_id)
        face_lst = get_faces(face_dir)
        if len(face_lst) == 0:
            all_h5f[utt_id] = np.zeros((1, 342))
            continue
        face_tensor = torch.from_numpy(get_face_tensors(face_lst)).to(device)
        utt_feat = []
        for face_tensor_bs in chunk(face_tensor, 32):
            extractor.set_input({"images": face_tensor_bs})
            extractor.forward()
            feat = extractor.out_ft.detach().cpu().numpy()
            utt_feat.append(feat)
        utt_feat = np.concatenate(utt_feat, axis=0)
        all_h5f[utt_id] = feat

def padding_to_fixlen(data, length):
    if len(data) >= length:
        ret = data[:length]
    else:
        ret = np.concatenate([data, np.zeros([length-len(data), data.shape[1]])], axis=0)
    return ret

def split_cv(config):
    max_len = 22
    all_ft = h5py.File(os.path.join(config['feature_root'], 'denseface_torch', 'all.h5'), "r")
    for cv in range(1, 11):
        save_dir = os.path.join(config['feature_root'], 'denseface_torch', str(cv))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for set_name in ['trn', 'val', 'tst']:
            int2name, _ = get_trn_val_tst(config['target_root'], cv, set_name)
            int2name = list(map(lambda x: x[0].decode(), int2name))
            fts = []
            for utt_id in int2name:
                ft = all_ft[utt_id][()]
                ft = padding_to_fixlen(ft, max_len)
                fts.append(ft)
            fts = np.array(fts)
            print(f'{cv} {set_name} {fts.shape}')
            np.save(os.path.join(save_dir, f'{set_name}.npy'), fts)

    
if __name__ == '__main__':
    config = {
        'face_root': '/data3/lrc/IEMOCAP',
        'feature_root': '/data7/lrc/IEMOCAP_features_npy/feature',
        'target_root': '/data7/lrc/IEMOCAP_features_npy/target'
    }
    model_cfg = {
        'model_name': 'densenet100',
        'num_blocks': 3,
        'growth_rate': 12, 
        'block_config': (16,16,16), 
        'init_kernel_size': 3,
        'num_init_features': 24, # growth_rate*2
        'reduction': 0.5,
        'bn_size': 4,
        'drop_rate': 0.0,
        'num_classes': 8,
        # train_params as below 
        'batch_size': 64,
        'max_epoch': 200,
        'optimizer': 'sgd',
        'nesterov': True,
        'momentum': 0.9,
        'weight_decay': 0,
        'learning_rate': 0.001,
        'reduce_half_lr_epoch': 40,
        'reduce_half_lr_rate': 0.5,  # epochs * 0.5
        'patience': 8,
        'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
        'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
        'data_augmentation': True,
        'validation_set': True,
        'validation_split': None,
        'num_threads': 4
    }
    # make_all_new_denseface(config)
    split_cv(config)