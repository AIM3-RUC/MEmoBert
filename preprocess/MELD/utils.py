import os
import os.path as osp
import json
import numpy as np

def get_config():
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), 'config.json')
    return json.load(open(config_path))

def get_int2name_label(config, set_name):
    # trn val tst 的int2name不唯一
    int2name_path = osp.join(config['target_dir'], set_name, 'int2name.npy')
    label_path = osp.join(config['target_dir'], set_name, 'label.npy')
    int2name = np.load(int2name_path)
    int2name = ["dia"+x[0].split('_')[0] + '_' + "utt"+x[0].split('_')[1] for x in int2name]
    label = np.load(label_path)
    return int2name, label

def find_video_using_uttid(config, set_name, utt_id):
    if set_name in ['train', 'trn']:
        return osp.join(config['data_dir'], 'train_splits', utt_id + ".mp4")
    elif set_name == 'val':
        return osp.join(config['data_dir'], 'dev_splits_complete', utt_id + ".mp4")
    elif set_name in ['test', 'tst']:
        return osp.join(config['data_dir'], 'output_repeated_splits_test', utt_id + ".mp4")

def make_or_exists(path):
    if not osp.exists(path):
        os.makedirs(path)