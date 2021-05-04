"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Dataset interfaces
"""
from contextlib import contextmanager
import io
import json
from os.path import exists

import numpy as np
from numpy.lib.function_base import copy
import copy
import torch
from torch import tensor
from torch.utils.data import Dataset, ConcatDataset
import horovod.torch as hvd
import lmdb
from lz4.frame import compress, decompress
from code.denseface.data.fer import augment_batch_images

import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor

import msgpack
import msgpack_numpy
msgpack_numpy.patch()


def _fp16_to_fp32(feat_dict):
    out = {k: arr.astype(np.float32)
           if arr.dtype == np.float16 else arr
           for k, arr in feat_dict.items()}
    return out

def _check_distributed():
    try:
        dist = hvd.size() != hvd.local_size()
    except ValueError:
        # not using horovod
        dist = False
    return dist

class DetectFeatLmdb(object):
    def __init__(self, img_dir, conf_th=0.2, max_bb=100, min_bb=10,
                 compress=True, data_augmentation=False):
        '''
        For both img and speech modalities
        '''
        # Jinming add: data_augmentation for raw image.
        self.data_augmentation = data_augmentation
        
        self.img_dir = img_dir
        # read the generated json file
        db_name = f'feat_th{conf_th}_max{max_bb}_min{min_bb}'
        nbb = f'nbb_th{conf_th}_max{max_bb}_min{min_bb}.json'
        print("[Info] Image db name {} data data_augmentation is {}".format(db_name, data_augmentation))
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
        # Jinming: now the input is rawimg, and the input is (64, 64)
        if len(img_dump['features'].shape) == 3:
            img_feat = img_dump['features'][:nbb, :, :]
            #Jinming: for data-augmentation
            if self.data_augmentation:
                # print('[Debug] before augment {} {}'.format(img_feat.shape, type(img_feat)))
                img_feat = augment_batch_images(img_feat)
                # print('[Debug] after augment {}'.format(img_feat.shape))
            img_feat = torch.tensor(img_feat).float()
        elif len(img_dump['features'].shape) == 2:
            img_feat = torch.tensor(img_dump['features'][:nbb, :].copy()).float()
        elif len(img_dump['features'].shape) == 1:
            # for raw speech signals
            img_feat = torch.tensor(img_dump['features'][:nbb].copy()).float()
        else:
            print("[Error] of img feature dimension {}".format(img_dump['features'].shape))
        return img_feat


@contextmanager
def open_lmdb(db_dir, readonly=False):
    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db

class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False,
                                 readahead=not _check_distributed())
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024**4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret


class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):
        if max_txt_len == -1:
            # 保持句子原有的长度
            self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        else:
            self.id2len = {
                id_: len_
                for id_, len_ in json.load(open(f'{db_dir}/id2len.json')
                                           ).items()
                if len_ <= max_txt_len
            }
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)
        meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump

    @property
    def txt2img(self):
        txt2img = json.load(open(f'{self.db_dir}/txt2img.json'))
        return txt2img

    @property
    def img2txts(self):
        img2txts = json.load(open(f'{self.db_dir}/img2txts.json'))
        return img2txts


def get_ids_and_lens(db):
    assert isinstance(db, TxtTokLmdb)
    lens = []
    ids = []
    # Modify by zjm: hvd.rank() is current process Id and hvd.size() is total gpus
    # for example, gpu0: 0~1/4L, gpu2: 1/4L~2/4L, gpu3: 2/4L~3/4L
    # Then the hvd.allgather() can restore original sequential order.
    splice_size = len(list(db.id2len.keys())) / hvd.size()
    start = int(hvd.rank() * splice_size)
    end = int(hvd.rank() * splice_size + splice_size)
    for id_ in list(db.id2len.keys())[start:end]:
        lens.append(db.id2len[id_])
        ids.append(id_)
    return lens, ids

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
            if img_shape == None:
                img_shape = (112, 112)
            img_feat = torch.zeros(img_shape).unsqueeze(0)
            num_bb = 1
        return img_feat, num_bb
    
    def _get_speech_feat(self, fname):
        # Jinimng: add this for speech 
        speech_feat = self.speech_db[fname]
        num_bb = speech_feat.size(0)
        return speech_feat, num_bb
    
    def _get_raw_speech_feat(self, fname):
        # Jinimng: audio speech features
        # https://github.com/huggingface/transformers/blob/b24ead87e1be6bce17e4ec5c953b6d028e4b3af7/src/transformers/models/wav2vec2/processing_wav2vec2.py#L77
        audio_path = self.speech_db[fname]
        speech = self.read_audio(audio_path)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        speech_feat = processor(speech, return_tensors="pt", sampling_rate=sr).input_values
        num_lens = speech_feat.size(0)
        return speech_feat, num_len
    
    def read_audio(wav_path):
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        if sr * 10 < len(speech):
            # print(f'{wav_path} long than 10 seconds and clip {speech.shape}')
            speech = speech[:int(sr * 10)]
        return speech, sr


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...], for 2d (batchsize, T, dim)
    Jinming add for 3d. (batchsize, T, 64, 64)
    """
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    if len(tensors[0].shape) == 3:
        output = torch.zeros(bs, max_len, hid, hid, dtype=dtype)
    elif len(tensors[0].shape) == 2:
        output = torch.zeros(bs, max_len, hid, dtype=dtype)
    elif len(tensors[0].shape) == 1:
        output = torch.zeros(bs, max_len, dtype=dtype)
    else:
        print('[Error] In pad_tensors is {}'.format(tensors.size))
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


class ConcatDatasetWithLens(ConcatDataset):
    """ A thin wrapper on pytorch concat dataset for lens batching """
    def __init__(self, datasets):
        super().__init__(datasets)
        self.lens = [l for dset in datasets for l in dset.lens]

    def __getattr__(self, name):
        return self._run_method_on_all_dsets(name)

    def _run_method_on_all_dsets(self, name):
        def run_all(*args, **kwargs):
            return [dset.__getattribute__(name)(*args, **kwargs)
                    for dset in self.datasets]
        return run_all

class ImageLmdbGroup(object):
    def __init__(self, conf_th, max_bb, min_bb, compress, data_augmentation=True):
        self.path2imgdb = {}
        self.conf_th = conf_th
        self.max_bb = max_bb
        self.min_bb = min_bb
        self.compress = compress
        self.data_augmentation = data_augmentation

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                    self.min_bb, self.compress,
                                    data_augmentation=self.data_augmentation)
        return img_db

class SpeechLmdbGroup(object):
    def __init__(self, speech_conf_th, max_frames, min_frames, compress, data_augmentation=False):
        self.path2imgdb = {}
        self.conf_th = speech_conf_th
        self.max_bb = max_frames
        self.min_bb = min_frames
        self.compress = compress
        self.data_augmentation = data_augmentation

    def __getitem__(self, path):
        img_db = self.path2imgdb.get(path, None)
        if img_db is None:
            img_db = DetectFeatLmdb(path, self.conf_th, self.max_bb,
                                    self.min_bb, self.compress,
                                    data_augmentation=self.data_augmentation)
        return img_db