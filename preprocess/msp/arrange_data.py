import os
import numpy as np
from sklearn import preprocessing

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_exist_face(uuid):
    face_root = '../../MSP-IMPROV_feature/face/raw'
    return os.path.exists(os.path.join(face_root, uuid+'.npy'))

def make_label():
    save_root = '../../MSP-IMPROV_feature/target'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    label_file = '../Evalution.txt'
    lines = open(label_file).readlines()
    records = list(filter(lambda x: '.avi' in x, lines))
    records = list(filter(lambda x: x.split('-')[4]=='S', records))
    int2name = []
    label = []
    label_set = ['A', 'H', 'N', 'S']
    no_face_count = 0
    no_face_record = []
    for record in records:
        uuid, label_name = record.split(';')[:2]
        uuid = uuid.strip().replace('UTD', 'MSP').split('.')[0]
        label_name = label_name.strip()
        if label_name in label_set and check_exist_face(uuid):
            int2name.append(uuid)
            label.append(label_set.index(label_name))
        else:
            print(uuid, label_name, check_exist_face(uuid))
            if not check_exist_face(uuid) and label_name in label_set:
                no_face_count += 1
                no_face_record.append(f'{uuid} {label_name} {check_exist_face(uuid)}\n')
            
    print("No face:", no_face_count)
    f = open('no_face_check.txt', 'w')
    f.writelines(no_face_record)
        
    int2name = np.array(int2name)
    label = np.array(label)
    int2name_path = os.path.join(save_root, 'all_int2name.npy')
    label_path = os.path.join(save_root, 'all_label.npy')
    print('int2name:', int2name.shape)
    print('label:', label.shape)
    np.save(int2name_path, int2name)
    np.save(label_path, label)

    for spk in ['M01', 'F01', 'M02', 'F02', 'M03', 'F03',
                'M04', 'F04', 'M05', 'F05', 'M06', 'F06']:
        spk_int2name = []
        spk_label = []
        for uuid, _label in zip(int2name, label):
            spkid = uuid.split('-')[3]
            if spkid == spk:
                spk_int2name.append(uuid)
                spk_label.append(_label)
        spk_int2name = np.array(spk_int2name)
        spk_label = np.array(spk_label)
        print(spk, 'int2name', spk_int2name.shape)
        print(spk, 'label', spk_label.shape)
        spk_int2name_path = os.path.join(save_root, f'{spk}_int2name.npy')
        spk_label_path = os.path.join(save_root, f'{spk}_label.npy')
        np.save(spk_int2name_path, spk_int2name)
        np.save(spk_label_path, spk_label)

def statistic():
    int2name = np.load('../../MSP-IMPROV_feature/target/all_int2name.npy')
    label = np.load('../../MSP-IMPROV_feature/target/all_label.npy')
    # face_frames
    face_feature_root = '../../MSP-IMPROV_feature/face/raw'
    face_lengths = []
    for uuid in int2name:
        face_feature_path = os.path.join(face_feature_root, uuid + '.npy')
        face_feature = np.load(face_feature_path)
        if len(face_feature) != 0:
            assert face_feature.shape[1] == 342
            face_lengths.append(face_feature.shape[0])
        else:
            print(uuid, 'has zero length')
        
    face_lengths.sort()
    _min = min(face_lengths)
    _max = max(face_lengths)
    mean = sum(face_lengths) / len(face_lengths)
    mid = face_lengths[int(len(face_lengths)/2)]
    sp_75 = face_lengths[int(len(face_lengths)*0.75)]
    print('Face:')
    print(f'Min:{_min} Max:{_max} Mean:{mean} Mid:{mid} 75%:{sp_75}')

    # word
    text_feature_root = '../../MSP-IMPROV_feature/text/raw'
    text_lengths = []
    for uuid in int2name:
        text_feature_path = os.path.join(text_feature_root, uuid + '.npy')
        text_feature = np.load(text_feature_path)
        if len(text_feature) != 0:
            assert text_feature.shape[1] == 1024
            text_lengths.append(text_feature.shape[0])
        
    text_lengths.sort()
    _min = min(text_lengths)
    _max = max(text_lengths)
    mean = sum(text_lengths) / len(text_lengths)
    mid = text_lengths[int(len(text_lengths)/2)]
    sp_75 = text_lengths[int(len(text_lengths)*0.75)]
    print('Text:')
    print(f'Min:{_min} Max:{_max} Mean:{mean} Mid:{mid} 75%:{sp_75}')

def statis_emo():
    all_label = np.load('../../MSP-IMPROV_feature/target/all_label.npy')
    record = {
        0:0, 1:0, 2:0, 3:0
    }
    for label in all_label:
        record[label] += 1
    print(len(all_label))
    print(record)

def gather():   
    face_len = 40
    text_len = 23
    for spk in ['M01', 'F01', 'M02', 'F02', 'M03', 'F03',
                'M04', 'F04', 'M05', 'F05', 'M06', 'F06']:

        int2name = np.load(f'../../MSP-IMPROV_feature/target/{spk}_int2name.npy')
        
        # audio
        save_path = f'../../MSP-IMPROV_feature/audio/IS10_{spk}.npy'
        feats = []
        for uuid in int2name:
            feat_file = '../../MSP-IMPROV_feature/audio/raw/' + uuid + '.npy'
            feat = np.load(feat_file)
            feats.append(feat)
        feats = np.array(feats)
        print('Audio total:', feats.shape)
        np.save(save_path, feats)

        # visual
        save_path = f'../../MSP-IMPROV_feature/face/denseface_{spk}.npy'
        feats = []
        for uuid in int2name:
            feat_file = '../../MSP-IMPROV_feature/face/raw/' + uuid + '.npy'
            feat = np.load(feat_file)
            if len(feat) >= face_len:
                feat = feat[:face_len]
            else:
                feat = np.concatenate([feat, np.zeros([face_len-len(feat), 342])])
            feats.append(feat)
        feats = np.array(feats)
        print('Visual total:', feats.shape)
        np.save(save_path, feats)

        # text
        save_path = f'../../MSP-IMPROV_feature/text/bert_{spk}.npy'
        feats = []
        for uuid in int2name:
            feat_file = '../../MSP-IMPROV_feature/text/raw/' + uuid + '.npy'
            feat = np.load(feat_file)
            if len(feat) >= text_len:
                feat = feat[:text_len]
            else:
                feat = np.concatenate([feat, np.zeros([text_len-len(feat), 1024])])
            feats.append(feat)
        feats = np.array(feats)
        print('Text total:', feats.shape)
        np.save(save_path, feats)


def make_cv_level_target():
    root = '/data6/lrc/MSP-IMPROV_feature/target/spk_level'
    save_root = '/data6/lrc/MSP-IMPROV_feature/target/cv_level'
    for cv in range(1, 13):
        val_gender = 'M' if cv % 2 == 1 else "F"
        val_num = (cv+1) // 2
        tst_gender = 'F' if cv % 2 == 1 else "M"
        tst_num = (cv+1) // 2
        val_spk = f'{val_gender}0{val_num}'
        tst_spk = f'{tst_gender}0{tst_num}'
        print('CV:', cv)
        print('val:', val_spk, 'tst:', tst_spk)
        trn_label, trn_int2name = [], []
        val_label, val_int2name = [], []
        tst_label, tst_int2name = [], []
        for spk in ['M01', 'F01', 'M02', 'F02', 'M03', 'F03',
                'M04', 'F04', 'M05', 'F05', 'M06', 'F06']:
            label = np.load(os.path.join(root, f'{spk}_label.npy'))
            int2name = np.load(os.path.join(root, f'{spk}_int2name.npy'))
            if spk == val_spk:
                val_label.append(label)
                val_int2name.append(int2name)
            elif spk == tst_spk:
                tst_label.append(label)
                tst_int2name.append(int2name)
            else:
                trn_label.append(label)
                trn_int2name.append(int2name)
        trn_label = np.concatenate(trn_label, axis=0)
        trn_int2name = np.concatenate(trn_int2name, axis=0)
        val_label = np.concatenate(val_label, axis=0)
        val_int2name = np.concatenate(val_int2name, axis=0)
        tst_label = np.concatenate(tst_label, axis=0)
        tst_int2name = np.concatenate(tst_int2name, axis=0)
        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        np.save(os.path.join(save_dir, 'trn_label.npy'), trn_label)
        np.save(os.path.join(save_dir, 'trn_int2name.npy'), trn_int2name)
        np.save(os.path.join(save_dir, 'val_label.npy'), val_label)
        np.save(os.path.join(save_dir, 'val_int2name.npy'), val_int2name)
        np.save(os.path.join(save_dir, 'tst_label.npy'), tst_label)
        np.save(os.path.join(save_dir, 'tst_int2name.npy'), tst_int2name)
        print('Trn:', trn_label.shape)
        print('Val:', val_label.shape)
        print('Tst:', tst_label.shape)
        assert(len(trn_label) == len(trn_int2name))
        assert(len(val_label) == len(val_int2name))
        assert(len(tst_label) == len(tst_int2name))

def make_cv_level_feature(modality):
    feat_name = {
        'audio': 'IS10',
        'face': 'denseface',
        'text': 'bert'
    }
    root = f'/data6/lrc/MSP-IMPROV_feature/{modality}/spk_level'
    save_root = f'/data6/lrc/MSP-IMPROV_feature/{modality}/cv_level'
    for cv in range(1, 13):
        val_gender = 'M' if cv % 2 == 1 else "F"
        val_num = (cv+1) // 2
        tst_gender = 'F' if cv % 2 == 1 else "M"
        tst_num = (cv+1) // 2
        val_spk = f'{val_gender}0{val_num}'
        tst_spk = f'{tst_gender}0{tst_num}'
        print('CV:', cv)
        print('val:', val_spk, 'tst:', tst_spk)
        trn_feat, val_feat, tst_feat = [], [], []
        for spk in ['M01', 'F01', 'M02', 'F02', 'M03', 'F03',
                'M04', 'F04', 'M05', 'F05', 'M06', 'F06']:
            feat = np.load(os.path.join(root, f'{feat_name[modality]}_{spk}.npy'))
            if spk == val_spk:
                val_feat.append(feat)
            elif spk == tst_spk:
                tst_feat.append(feat)
            else:
                trn_feat.append(feat)
        trn_feat = np.concatenate(trn_feat, axis=0)
        val_feat = np.concatenate(val_feat, axis=0)
        tst_feat = np.concatenate(tst_feat, axis=0)
        if modality == 'audio':
            scaler = preprocessing.StandardScaler().fit(trn_feat)
            trn_feat = scaler.transform(trn_feat)
            val_feat = scaler.transform(val_feat)
            tst_feat = scaler.transform(tst_feat)

        save_dir = os.path.join(save_root, str(cv))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        np.save(os.path.join(save_dir, 'trn.npy'), trn_feat)
        np.save(os.path.join(save_dir, 'val.npy'), val_feat)
        np.save(os.path.join(save_dir, 'tst.npy'), tst_feat)
        print('Trn:', trn_feat.shape)
        print('Val:', val_feat.shape)
        print('Tst:', tst_feat.shape)


def make_miss_modality_mix_data(src, tgt, label_dir, new_label_dir, cv=1, phase='val'):
    cv = str(cv)
    A = np.load(os.path.join(src, 'audio/cv_level', cv, phase + '.npy'))
    V = np.load(os.path.join(src, 'face/cv_level', cv, phase + '.npy'))
    L = np.load(os.path.join(src, 'text/cv_level', cv, phase + '.npy'))
    LABEL = np.load(os.path.join(label_dir, cv, phase + '_label.npy'))
    INT2NAME = np.load(os.path.join(label_dir, cv, phase + '_int2name.npy'))
    new_A = []
    new_V = []
    new_L = []
    new_label = []
    new_int2name = []
    miss_type = []
    modalities = ['a', 'v', 'l']
    for a, v, l, label, int2name in zip(A, V, L, LABEL, INT2NAME):
       a = np.expand_dims(a, 0)
       v = np.expand_dims(v, 0)
       l = np.expand_dims(l, 0)
       # A + Z + Z
       new_A.append(a)
       new_V.append(np.zeros(v.shape))
       new_L.append(np.zeros(l.shape))
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('azz')
       # Z + V + Z
       new_A.append(np.zeros(a.shape))
       new_V.append(v)
       new_L.append(np.zeros(l.shape))
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('zvz')
       # Z + Z + L
       new_A.append(np.zeros(a.shape))
       new_V.append(np.zeros(v.shape))
       new_L.append(l)
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('zzl')
       # A + V + Z
       new_A.append(a)
       new_V.append(v)
       new_L.append(np.zeros(l.shape))
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('avz')
       # A + Z + L
       new_A.append(a)
       new_V.append(np.zeros(v.shape))
       new_L.append(l)
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('azl')
       # Z + V + L
       new_A.append(np.zeros(a.shape))
       new_V.append(v)
       new_L.append(l)
       new_label.append(label)
       new_int2name.append(int2name)
       miss_type.append('zvl')

    new_A = np.vstack(new_A)
    new_V = np.vstack(new_V)
    new_L = np.vstack(new_L)
    new_label = np.vstack(new_label)
    new_int2name = np.array(new_int2name)
    miss_type = np.array(miss_type)
    save_a = os.path.join(tgt, 'audio/miss', cv)
    save_v = os.path.join(tgt, 'face/miss', cv)
    save_l = os.path.join(tgt, 'text/miss', cv)
    save_target = os.path.join(new_label_dir, cv)
    mkdir(save_a)
    mkdir(save_v)
    mkdir(save_l)
    mkdir(save_target)
    save_a = os.path.join(save_a, phase + '.npy')
    save_l = os.path.join(save_l, phase + '.npy')
    save_v = os.path.join(save_v, phase + '.npy')
    save_label = os.path.join(save_target, phase + '_label.npy')
    save_int2name = os.path.join(save_target, phase + '_int2name.npy')
    save_miss_type = os.path.join(save_target, phase + '_type.npy')
    print('Save to ' + save_a)
    print('Save to ' + save_v)
    print('Save to ' + save_l)
    print('Save to ' + save_label)
    print('Save to ' + save_int2name)
    print('Save to ' + save_miss_type)
    np.save(save_a, new_A)
    np.save(save_v, new_V)
    np.save(save_l, new_L)
    np.save(save_label, new_label)
    np.save(save_int2name, new_int2name)
    np.save(save_miss_type, miss_type)


def check_data(root):
    cv = '3'
    phase = 'val'
    A = np.load(os.path.join(root, 'audio/miss', cv, phase + '.npy'))
    V = np.load(os.path.join(root, 'face/miss', cv, phase + '.npy'))
    L = np.load(os.path.join(root, 'text/miss', cv, phase + '.npy'))
    print(os.path.join(root, 'audio/miss', cv, phase + '.npy'))
    print(os.path.join(root, 'face/miss', cv, phase + '.npy'))
    print(os.path.join(root, 'text/miss', cv, phase + '.npy'))
    input()
    for a, v, l in zip(A, V, L):
        print(a.shape, np.sum(a))
        print(v.shape, np.sum(v))
        print(l.shape, np.sum(l))
        input()

# make_label()
# statistic()
# gather()
# make_cv_level_target()
# make_cv_level_feature('audio')
# make_cv_level_feature('face')
# make_cv_level_feature('text')

# src = '/data6/lrc/MSP-IMPROV_feature/'
# tgt = '/data6/lrc/MSP-IMPROV_feature/'
# label_dir = '/data6/lrc/MSP-IMPROV_feature/target/cv_level'
# new_label_dir = '/data6/lrc/MSP-IMPROV_feature/target/miss'
# for cv in range(1, 13):
#     for phase in ['val', 'tst']:
#         make_miss_modality_mix_data(src, tgt, label_dir, new_label_dir, cv, phase)
# check_data('/data6/lrc/MSP-IMPROV_feature/')

statis_emo()