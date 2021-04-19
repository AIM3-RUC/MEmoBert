import time
import os, shutil
import glob
import numpy as np

def read_IS10(fname):
    content = open(fname).readlines()
    data_str = next(filter(lambda x: 'unknown' in x, content))
    data = np.array(data_str.split(',')[1: -1], dtype=np.float32)
    return data


def extract_IS10(wav_name):
    tmp_dir = '.tmp/IS10'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    uuid = os.path.basename(wav_name).split('.')[0]
    out_path = os.path.join(tmp_dir, '{}_IS10.csv'.format(uuid))
    if not os.path.exists(out_path):
        cmd = 'SMILExtract -C /root/opensmile-2.3.0/config/IS10_paraling.conf -I {} -O {} -noconsoleoutput 1'.format(wav_name, out_path)
        os.system(cmd)
    feature = read_IS10(out_path)
    return feature

def make_all_IS10():
    audio_root = '../Audio'
    save_root = '../../MSP-IMPROV_feature/audio/raw'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    all_wavs = glob.glob(os.path.join(audio_root, '*/*/S/*.wav'))
    for wav in all_wavs:
        feat = extract_IS10(wav)
        uuid = uuid = os.path.basename(wav).split('.')[0]
        print(uuid, feat.shape)
        save_path = os.path.join(save_root, uuid+'.npy')
        np.save(save_path, feat)


if __name__ == '__main__':
    make_all_IS10()