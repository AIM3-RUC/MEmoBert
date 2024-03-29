import os
import h5py
import json
import numpy as np
import pandas as pd
import scipy.signal as spsig
from tqdm import tqdm


class ComParEExtractor(object):
    ''' 抽取comparE特征, 输入音频路径, 输出npy数组, 每帧130d
    '''
    def __init__(self, opensmile_tool_dir=None, downsample=10, tmp_dir='.tmp', no_tmp=False):
        ''' Extract ComparE feature
            tmp_dir: where to save opensmile csv file
            no_tmp: if true, delete tmp file
        '''
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if opensmile_tool_dir is None:
            opensmile_tool_dir = '/root/opensmile-2.3.0/'
        self.opensmile_tool_dir = opensmile_tool_dir
        self.tmp_dir = tmp_dir
        self.downsample = downsample
        self.no_tmp = no_tmp
    
    def __call__(self, wav):
        basename = os.path.basename(wav).split('.')[0]
        save_path = os.path.join(self.tmp_dir, basename+".csv")
        cmd = 'SMILExtract -C {}/config/ComParE_2016.conf \
            -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1 \
            -I {} -lldcsvoutput {} -instname xx -O ? -noconsoleoutput 1'
        os.system(cmd.format(self.opensmile_tool_dir, wav, save_path))
        
        df = pd.read_csv(save_path, delimiter=';')
        wav_data = df.iloc[:, 2:]
        if len(wav_data) > self.downsample:
            wav_data = spsig.resample_poly(wav_data, up=1, down=self.downsample, axis=0)
            if self.no_tmp:
                os.remove(save_path) 
        else:
            wav_data = None
            self.print(f'Error in {wav}, no feature extracted')

        return wav_data

def transform_utt_id(utt_id, set_name):
    dia_num, utt_num = utt_id.split('_')
    return f'{set_name}/dia{dia_num}_utt{utt_num}'

def get_trn_val_tst(target_root_dir, setname):
    int2name = np.load(os.path.join(target_root_dir, setname, 'int2name.npy'))
    int2label = np.load(os.path.join(target_root_dir, setname, 'label.npy'))
    int2name = [transform_utt_id(utt_id, setname) for utt_id in int2name[:, 0].tolist()]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_all_comparE(config):
    extractor = ComParEExtractor()
    trn_int2name, _ = get_trn_val_tst(config['target_dir'], 'train')
    val_int2name, _ = get_trn_val_tst(config['target_dir'], 'val')
    tst_int2name, _ = get_trn_val_tst(config['target_dir'], 'test')
    all_utt_ids = trn_int2name + val_int2name + tst_int2name
    all_h5f = h5py.File(os.path.join(config['output_dir'], 'feature', 'comparE', 'all.h5'), 'w')
    for utt_id in tqdm(all_utt_ids):
        wav_path = os.path.join(config['output_dir'], 'audio', utt_id + '.wav')
        feat = extractor(wav_path)
        all_h5f[utt_id] = feat

def normlize_on_trn(config, input_file, output_file):
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    for cv in range(1, 11):
        trn_int2name, _ = get_trn_val_tst(config['target_dir'], cv, 'trn')
        trn_int2name = list(map(lambda x: x[0].decode(), trn_int2name))
        all_feat = [in_data[utt_id][()] for utt_id in trn_int2name]
        all_feat = np.concatenate(all_feat, axis=0)
        mean_f = np.mean(all_feat, axis=0)
        std_f = np.std(all_feat, axis=0)
        std_f[std_f == 0.0] = 1.0
        cv_group = h5f.create_group(str(cv))
        cv_group['mean'] = mean_f
        cv_group['std'] = std_f
        print(cv)
        print("mean:", np.sum(mean_f))
        print("std:", np.sum(std_f))
    
    
if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    config_path = os.path.join(pwd, '../', 'config.json')
    config = json.load(open(config_path))
    make_all_comparE(config)
    # normlize_on_trn(config, os.path.join(config['output_dir'], 'feature', 'comparE', 'all.h5'), os.path.join(config['output_dir'], 'comparE', 'mean_std.h5'))