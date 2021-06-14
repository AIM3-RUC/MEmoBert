import json
import lmdb
import os
from tqdm import tqdm
from lz4.frame import compress, decompress
import numpy as np
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

'''
将两个不同特征的db的数据合并起来，比如目前的 wav2vec 和 wav2vec_cnn
/data7/MEmoBert/emobert/wav2vec_db/movies_v3_3mean/
/data7/MEmoBert/emobert/wav2vec_db/movies_v3_cnn_3mean/
'''

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

def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)

if __name__ == '__main__':

    # ft1_db_dir = '/data7/emobert/wav2vec_db/movies_v3_3mean/feat_th1.0_max64_min10/'
    # ft1_filepath = '/data7/emobert/wav2vec_db/movies_v3_3mean/nbb_th1.0_max64_min10.json'
    # ft2_db_dir = '/data7/emobert/wav2vec_db/movies_v3_cnn_3mean/feat_th1.0_max64_min10/'
    # save_db_dir = '/data7/emobert/wav2vec_db/movies_v3_globalcnn_3mean/feat_th1.0_max64_min10/'
    # save_filepath = '/data7/emobert/wav2vec_db/movies_v3_globalcnn_3mean/nbb_th1.0_max64_min10.json'

    ft1_db_dir = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_db_3mean/feat_th1.0_max64_min10/'
    ft1_filepath = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_db_3mean/nbb_th1.0_max64_min10.json'
    ft2_db_dir = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_cnn_db_3mean/feat_th1.0_max64_min10/'
    save_db_dir = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_globalcnn_db_3mean/feat_th1.0_max64_min10/'
    save_filepath = '/data7/emobert/exp/evaluation/MSP/feature/wav2vec_globalcnn_db_3mean/nbb_th1.0_max64_min10.json'
    
    if not os.path.exists(save_db_dir):
        os.makedirs(save_db_dir)

    env1 = lmdb.open(ft1_db_dir,
                    readonly=True, create=False)
    txn1 = env1.begin(buffers=True)
    env2 = lmdb.open(ft2_db_dir,
                    readonly=True, create=False)
    txn2 = env2.begin(buffers=True)

    # save the new features
    env = lmdb.open(save_db_dir, map_size=1024**4)
    txn = env.begin(write=True)

    i = 0 
    video2lens = json.load(open(ft1_filepath))
    for file_name in tqdm(video2lens.keys()):
        i += 1
        new_dump = {}
        dump1 = txn1.get(file_name.encode('utf-8'))
        img_dump1 = msgpack.loads(dump1, raw=False)
        dump2 = txn2.get(file_name.encode('utf-8'))
        img_dump2 = msgpack.loads(dump2, raw=False)

        # print(img_dump1['features'].shape, img_dump2['features'].shape)
        assert img_dump1['features'].shape[0] == img_dump2['features'].shape[0]
        new_fts = np.concatenate([img_dump1['features'], img_dump2['features']], axis=-1)
        # print(new_fts.shape)
        new_dump['features'] = new_fts
        dump = dumps_msgpack(new_dump)
        txn.put(key=file_name.encode('utf-8'), value=dump)
        if i % 1000 == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.put(key=b'__keys__',
            value=json.dumps(list(video2lens.keys())).encode('utf-8'))
    txn.commit()
    env.close()
    with open(save_filepath, 'w') as f:
        json.dump(video2lens, f)