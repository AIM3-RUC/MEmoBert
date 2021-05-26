"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""
from contextlib import contextmanager
import io
import copy
import json
from os.path import exists

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import horovod.torch as hvd
import lmdb
from lz4.frame import compress, decompress

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

# from uniter 
from code.uniter.data.data import TxtLmdb, TxtTokLmdb, \
        get_ids_and_lens, _check_distributed, _fp16_to_fp32, pad_tensors, ConcatDatasetWithLens

@contextmanager
def open_lmdb(db_dir, readonly=False):
    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db

class DetectFeatLmdb(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10, compress=False):
        # read the generated json file
        db_name = img_dir.split('/')[-1]
        nbb = db_name.replace('feat', 'nbb') + '.json'
        print(db_name, nbb)
        img_dir = img_dir.replace(db_name, '')
        print(img_dir)
        print("[Debug] Loading Image db {}".format(db_name))
        if not exists(f'{img_dir}/{nbb}'):
            print('[Error]: nbb is not pre-computed and the json-file may be error!')
            self.name2nbb = None
        else:
            self.name2nbb = json.load(open(f'{img_dir}/{nbb}'))
        self.compress = compress
        if compress:
            db_name += '_compressed'

        # only read ahead on single node training
        self.env = lmdb.open(f'{img_dir}/{db_name}',
                             readonly=True, create=False,
                             readahead=not _check_distributed())
        self.txn = self.env.begin(buffers=True)
        print("[Debug] Loading Image db {} Done!!!".format(db_name))

    def __del__(self):
        self.env.close()

    def get_dump(self, file_name):
        # hack for MRC
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = _fp16_to_fp32(img_dump)
        else:
            img_dump = msgpack.loads(dump, raw=False)
            img_dump = _fp16_to_fp32(img_dump)
        img_dump = {k: arr[:nbb, ...] for k, arr in img_dump.items()}
        return img_dump

    def __getitem__(self, file_name):
        # Jinming, no norm-bbx feature and only position ids is OK.
        dump = self.txn.get(file_name.encode('utf-8'))
        nbb = self.name2nbb[file_name]
        if self.compress:
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                img_dump = {'features': img_dump['features']}
        else:
            img_dump = msgpack.loads(dump, raw=False)
        img_feat = torch.tensor(np.array(img_dump['features'])[:nbb, :]).float()
        return img_feat

class DetectFeatTxtTokDataset(Dataset):
    def __init__(self, txt_db, img_db=None, speech_db=None):
        '''
        所有的数据检索都以以txt作为标准，所以txt_db不能为None, 可以通过其他方法进行控制
        '''
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db:
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db:
            assert isinstance(speech_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.lens = copy.deepcopy(self.txt_lens)
        txt2img = txt_db.txt2img
        if img_db is not None:
            print('[Debug in data] add the img lens')
            self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(self.lens, self.ids)]
        if speech_db is not None:
            print('[Debug in data] add the speech lens')
            self.lens = [tl + self.speech_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(self.lens, self.ids)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        return example

    def _get_img_feat(self, fname, img_shape):
        # Jinimng: remove the norm-bbx features 
        img_feat = self.img_db[fname]
        num_bb = img_feat.size(0)
        if num_bb == 0:
            # print('[Warn] current {} is empty img info!!!\n'.format(fname))
            if img_shape == None:
                # Jinming: add for the first sample is none
                print("[Warning] Set the img_shape to 342!!!")
                img_shape = 342
            img_feat = torch.zeros(img_shape).unsqueeze(0)
            num_bb = 1
        # print('[Info] img_feat shape {}'.format(img_feat.shape))
        return img_feat, num_bb

    def _get_speech_feat(self, fname):
        # Jinimng: add this for speech 
        speech_feat = self.speech_db[fname]
        num_bb = speech_feat.size(0)
        return speech_feat, num_bb

def get_gather_index(txt_lens, num_bbs, num_frames, batch_size, max_len, out_size):
    '''
    Jinming modify this for multimodalies.
    '''
    assert len(txt_lens) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    if num_bbs is not None and num_frames is None:
        for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
            gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                        dtype=torch.long).data
    elif num_bbs is None and num_frames is not None:
        for i, (tl, nbb) in enumerate(zip(txt_lens, num_frames)):
            gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                        dtype=torch.long).data
    elif num_bbs is not None and num_frames is not None:
        max_bb = max(num_bbs)
        for i, (tl, nbb, nframe) in enumerate(zip(txt_lens, num_bbs, num_frames)):
            # a bug 
            gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                        dtype=torch.long).data
            gather_index.data[i, tl+nbb:tl+nbb+nframe] = torch.arange(max_len+max_bb, max_len+max_bb+nframe,
                                                        dtype=torch.long).data
    elif num_bbs is None and num_frames is None:
        # 如果只包含文本信息的话，那么不需要进行gather.
        pass
    else:
        print('[Error] Error in gather index')
    return gather_index

def get_gather_index_notxtdb(num_bbs, num_frames, batch_size, max_bb, out_size):
    '''
    Jinming modify this for multimodalies.
    '''
    gather_index = torch.arange(0, out_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    if num_bbs is None or num_frames is None:
        # 如果只包含一个模态，那么不需要做gather-index
       pass
    else:
        # 同时包含图片和语音模态
        for i, (nbb, nframe) in enumerate(zip(num_bbs, num_frames)):
            gather_index.data[i, nbb:nbb+nframe] = torch.arange(max_bb, max_bb+nframe,
                                                        dtype=torch.long).data
    return gather_index

class ImageLmdbGroup(object):
    def __init__(self, compress):
        self.path2imgdb = {}
        self.compress = compress

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.compress)
        return img_db

class SpeechLmdbGroup(object):
    def __init__(self, compress):
        self.path2imgdb = {}
        self.compress = compress

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.compress)
        return img_db