import json
import lmdb
from lz4.frame import compress, decompress
import numpy as np
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
           if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out

def read_img_lmdb(lmdb_dir, file_name='test-dia162_utt0.npz'):
    # only read ahead on single node training
    env = lmdb.open(lmdb_dir,
                    readonly=True, create=False)
    txn = env.begin(buffers=True)
    dump = txn.get(file_name.encode('utf-8'))
    img_dump = msgpack.loads(dump, raw=False)
    img_dump = _fp16_to_fp32(img_dump)
    nbb = 20
    img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
    return img_dump

def compute_stastic_info(lens):
    # 返回长度的中位数和80%分位点
    lens.sort()
    avg_len = sum(lens) / len(lens)
    mid_len = lens[int(len(lens)/2)]
    m80_len = lens[int(len(lens)*0.8)]
    return avg_len, mid_len, m80_len

<<<<<<< HEAD
# lmdb_name = 'movies_v2/fc'
# filepath = '/data7/emobert/img_db_nomask/{}/nbb_th0.5_max36_min10.json'.format(lmdb_name)
# filepath = '/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/nbb_th0.0_max36_min10.json'
filepath = '/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/img_db/fc/nbb_th0.0_max64_min10.json'

video2lens = json.load(open(filepath))
movie_faces_lens = []
for key in video2lens.keys():
    _len = video2lens[key]
    movie_faces_lens.append(_len)
avg_len, min_len, m80_len = compute_stastic_info(movie_faces_lens)
# print(movie_faces_lens[0:30])
# print(movie_faces_lens[540:600])
print('\t Face {} Avg {:.2f} Mid {:.2f} Mid80 {:.2f}'.format(len(movie_faces_lens), avg_len, min_len, m80_len) + '\n')
=======

if __name__ == '__main__':
    lmdb_name = 'movies_v2/fc'
    filepath = '/data7/emobert/img_db_nomask/{}/nbb_th0.5_max36_min10.json'.format(lmdb_name)
    # filepath = '/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_iemocap_mean_std_torch/img_db/{}/nbb_th0.1_max64_min10.json'.format(lmdb_name)

    if False:
        video2lens = json.load(open(filepath))
        movie_faces_lens = []
        for key in video2lens.keys():
            _len = video2lens[key]
            movie_faces_lens.append(_len)
        avg_len, min_len, m80_len = compute_stastic_info(movie_faces_lens)
        print('\t Face {} Avg {:.2f} Mid {:.2f} Mid80 {:.2f}'.format(len(movie_faces_lens), avg_len, min_len, m80_len) + '\n')
    
    if True:
        lmdb_name_dir = '/data4/MEmoBert/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/img_db/fc/feat_th0.1_max36_min10'
        img_dump = read_img_lmdb(lmdb_name_dir)
        print(img_dump['soft_labels'].shape)
        # meld 的 soft-label 不对
>>>>>>> ef69baa4de2ef7334d4a0672b688e073d90c818c
