import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from glob import glob
from preprocess.MELD.utils import get_config, get_int2name_label, make_or_exists

def transform_utt_id(utt_id, set_name):
    dia_num, utt_num = utt_id.split('_')
    return f'{set_name}/dia{dia_num}_utt{utt_num}'

def get_all_utt_ids():
    target_root = '/data7/MEmoBert/evaluation/MELD/target'
    ans = []
    for set_name in ['train', 'val', 'test']:
        utt_ids = np.load(os.path.join(target_root, set_name, f'int2name.npy'))
        utt_ids = [transform_utt_id(utt_id, set_name) for utt_id in utt_ids[:, 0].tolist()]
        ans += utt_ids
    return ans


def make_src_tgt(config=None, src='./src.lst', tgt='./tgt.lst'):
    if config is None:
        config = get_config()
    utt_ids = get_all_utt_ids()
    frame_dir = osp.join(config['output_dir'], 'frames')
    save_dir = osp.join(config['output_dir'], 'seetafaces')
    make_or_exists(save_dir)
    src = open(src, 'w', encoding='utf8')
    tgt = open(tgt, 'w', encoding='utf8')
    for utt_id in tqdm(utt_ids):
        sub_dir = osp.join(save_dir, utt_id)
        make_or_exists(sub_dir)
        utt_frames = sorted(glob(osp.join(frame_dir, utt_id, '*.jpg')), 
                    key=lambda x: int(osp.basename(x).split('.')[0]))
        utt_faces = [x.replace('frames', 'seetafaces') for x in utt_frames]
        utt_frames = [x + '\n' for x in utt_frames]
        utt_faces = [x + '\n' for x in utt_faces]
        src.writelines(utt_frames)
        tgt.writelines(utt_faces)

if __name__ == '__main__':
    make_src_tgt()

'''
export LD_LIBRARY_PATH=/data3/lrc/IEMOCAP_full_release/code:/root/anaconda2/pkgs/libopencv-3.4.2-hb342d67_1/lib/libopencv_core.so.3.4:$LD_LIBRARY_PATH
/data2/zjm/tools/seetface/seetaface_detection  /data2/zjm/tools/seetface/seeta_fd_frontal_v1.0.bin  \
                src.lst  \
                tgt.lst  0 635454
'''