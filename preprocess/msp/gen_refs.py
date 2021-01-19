import json
import os
import numpy as np
from os.path import join

'''
ref fromat:
{
    'segmentId': {'label': 1}
}
'''

def get_trn_val_tst(target_root_dir, cv, setname):
    int2name = np.load(join(target_root_dir, cv, '{}_int2name.npy'.format(setname)))
    int2label = np.load(join(target_root_dir, cv, '{}_label.npy'.format(setname)))
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_ref(int2name, labels, txt_root, save_path):
    ref = {}
    for i in range(len(int2name)):
        utt_id = str(int2name[i])
        label = str(labels[i])
        txt_path = os.path.join(txt_root, utt_id + '.txt')
        txt = open(txt_path).read().strip()
        ref[utt_id] = {
            'txt': [txt],
            'label': label
        }
    json.dump(ref, open(save_path, 'w', encoding='utf8'), indent=4)

if __name__ == '__main__':
    root_dir = '/data7/MEmoBert/emobert/exp/evaluation/MSP-IMPROV'
    txt_root = '/data6/lrc/MSP-IMPROV/All_human_transcriptions/'
    target_root_dir = os.path.join(root_dir, 'target')
    save_root_dir = os.path.join(root_dir, 'refs')
    for cv in range(1, 13):
        save_dir = os.path.join(save_root_dir, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trn_int2name, trn_label = get_trn_val_tst(target_root_dir, str(cv),  'trn')
        val_int2name, val_label = get_trn_val_tst(target_root_dir, str(cv), 'val')
        tst_int2name, tst_label = get_trn_val_tst(target_root_dir, str(cv), 'tst')
        save_path_trn = os.path.join(save_dir, 'trn_ref.json')
        save_path_val = os.path.join(save_dir, 'val_ref.json')
        save_path_tst = os.path.join(save_dir, 'tst_ref.json')
        make_ref(trn_int2name, trn_label, txt_root, save_path_trn)
        make_ref(val_int2name, val_label, txt_root, save_path_val)
        make_ref(tst_int2name, tst_label, txt_root, save_path_tst)