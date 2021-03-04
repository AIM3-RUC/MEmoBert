import json
import os
import numpy as np
from glob import glob

def make_ref(utt_ids, labels, txt_info, save_path):
    ans = {}
    for utt_id, label in zip(utt_ids, labels):
        ans[utt_id] = {
            'txt': [txt_info[utt_id]],
            'label': label
        }
    json.dump(ans, open(save_path, 'w', encoding='utf8'), indent=4)

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

def process_sentence(sentence):
    sentence = ' '.join(sentence.split(':')[1:]).strip()
    return sentence

def get_sentence_map(file):
    lines = open(file).readlines()
    lines = list(filter(lambda x: not(x.startswith('M:') or x.startswith('F:')), lines))
    ans = list(map(lambda x: (x.split()[0], process_sentence(x)), lines))
    return dict(ans)
    
def get_session_map(session):
    items = []
    root = "/data3/zjm/dataset/IEMOCAP_full_release/"
    txt_files = glob(os.path.join(root, f'Session{session}/dialog/transcriptions/*.txt'))
    for txt_file in txt_files:
        items += list(get_sentence_map(txt_file).items())
    return dict(items)

def get_txt_infos():
    ans = {}
    for i in range(1, 6):
        sess_map = get_session_map(i)
        ans = {**ans, **sess_map}
    return ans

if __name__ == '__main__':
    save_root = 'refs'
    txt_info = get_txt_infos()
    for cv in range(1, 11):
        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        trn_int2name, val_int2name, tst_int2name, trn_label, val_label, tst_label = get_trn_val_tst(cv)
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name.tolist()))
        val_int2name = list(map(lambda x: x[0].decode(), val_int2name.tolist()))
        tst_int2name = list(map(lambda x: x[0].decode(), tst_int2name.tolist()))
        trn_label = np.argmax(trn_label, axis=1).tolist()
        val_label = np.argmax(val_label, axis=1).tolist()
        tst_label = np.argmax(tst_label, axis=1).tolist()
        save_path_trn = os.path.join(save_dir, 'trn_ref.json')
        save_path_val = os.path.join(save_dir, 'val_ref.json')
        save_path_tst = os.path.join(save_dir, 'tst_ref.json')
        make_ref(trn_int2name, trn_label, txt_info, save_path_trn)
        make_ref(val_int2name, val_label, txt_info, save_path_val)
        make_ref(tst_int2name, tst_label, txt_info, save_path_tst)
