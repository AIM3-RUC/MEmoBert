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
from code.uniter.data.data import DetectFeatLmdb, TxtLmdb, TxtTokLmdb, \
        get_ids_and_lens, pad_tensors, ConcatDatasetWithLens

@contextmanager
def open_lmdb(db_dir, readonly=False):
    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db

class DetectFeatTxtTokDataset(Dataset):
    def __init__(self, txt_db, img_db=None, speech_db=None):
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
        if img_db:
            self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(self.lens, self.ids)]
        if speech_db:
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
    return gather_index

class ImageLmdbGroup(object):
    def __init__(self, conf_th, max_bb, min_bb, compress):
        self.path2imgdb = {}
        self.conf_th = conf_th
        self.max_bb = max_bb
        self.min_bb = min_bb
        self.compress = compress

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                    self.min_bb, self.compress)
        return img_db

class SpeechLmdbGroup(object):
    def __init__(self, speech_conf_th, max_frames, min_frames, compress):
        self.path2imgdb = {}
        self.conf_th = speech_conf_th
        self.max_bb = max_frames
        self.min_bb = min_frames
        self.compress = compress

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                    self.min_bb, self.compress)
        return img_db