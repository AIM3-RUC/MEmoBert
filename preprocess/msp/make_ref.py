import os, shutil
import glob
import json
import numpy as np
from tqdm import tqdm

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

def get_text(utt_id):
    transcript_root = '/data6/lrc/MSP-IMPROV/All_human_transcriptions'
    return open(os.path.join(transcript_root, utt_id + '.txt')).read()

def make_ref():
    save_dir = '/data7/emobert/exp/evaluation/MSP-IMPROV/feature/text'
    for cv in range(1, 13):
        trn_int2name, val_int2name, tst_int2name, trn_label, val_label, tst_label \
            = get_trn_val_tst(cv, '/data6/lrc/MSP-IMPROV_feature/target/cv_level')
        
        # for 11:1 setting
        trn_int2name = np.concatenate([trn_int2name, tst_int2name])
        tst_int2name = val_int2name
        trn_label = np.concatenate([trn_label, tst_label])
        tst_label = val_label

        trn_ref, val_ref, tst_ref = {}, {}, {}
        cv_save_dir = os.path.join(save_dir, str(cv))
        if not os.path.exists(cv_save_dir):
            os.mkdir(cv_save_dir)

        for utt_id, label in zip(trn_int2name, trn_label):
            trn_ref[utt_id] = {
                'txt': [get_text(utt_id)],
                'label': int(label)
            }
        json.dump(trn_ref, open(os.path.join(cv_save_dir, 'trn.json'), 'w', encoding='utf8'), indent=4)

        for utt_id, label in zip(val_int2name, val_label):
            val_ref[utt_id] = {
                'txt': [get_text(utt_id)],
                'label': int(label)
            }
        json.dump(val_ref, open(os.path.join(cv_save_dir, 'val.json'), 'w', encoding='utf8'), indent=4)

        for utt_id, label in zip(tst_int2name, tst_label):
            tst_ref[utt_id] = {
                'txt': [get_text(utt_id)],
                'label': int(label)
            }
        json.dump(tst_ref, open(os.path.join(cv_save_dir, 'tst.json'), 'w', encoding='utf8'), indent=4)

def cp_wavs():
    audio_root = '/data6/lrc/MSP-IMPROV/Audio/'
    target_dir = '/data7/MEmoBert/emobert/exp/evaluation/MSP/audio'
    trn_int2name, val_int2name, tst_int2name, _, _, _ = \
        get_trn_val_tst(1, '/data6/lrc/MSP-IMPROV_feature/target/cv_level')
    int2name = trn_int2name.tolist() + val_int2name.tolist() + tst_int2name.tolist()
    for utt_id in tqdm(int2name):
        sess_id = utt_id.split('-')[3][2]
        dialog_id = utt_id.split('-')[2]
        wav_path = f'{audio_root}/session{sess_id}/{dialog_id}/S/{utt_id}.wav'
        tgt_path = f'{target_dir}/{utt_id}.wav'
        shutil.copyfile(wav_path, tgt_path)
    
# make_ref()
cp_wavs()