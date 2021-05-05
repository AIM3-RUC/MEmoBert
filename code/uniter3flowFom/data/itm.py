"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
import math
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from code.uniter3flowFom.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_ids_and_lens)
from code.uniter3flowFom.data.sampler import TokenBucketSampler
from code.uniter3m.data.data import get_gather_index

class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs

class ItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) 
    不能直接继承 DetectFeatTxtTokDataset 的 init 函数。
    """
    def __init__(self, txt_db, img_db, speech_db, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db:
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db:
            assert isinstance(speech_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.txt_lens, self.ids = get_ids_and_lens(txt_db)

        self.img_shape = None

        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1-self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            if self.labels[i] == 0:
                img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
            self.train_imgs.append(img_fname)
            if self.img_db and self.speech_db is None:
                self.lens.append(tl + self.img_db.name2nbb[img_fname])
            if self.speech_db and self.img_db is None :
                self.lens.append(tl + self.speech_db.name2nbb[img_fname])
            if self.speech_db and self.img_db:
                self.lens.append(tl + self.speech_db.name2nbb[img_fname] + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        '''
        由于speech 和 face 的 imageId 都是一样的, 所以只需要修改这里就可以了
        '''
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)
        img_fname = example['img_fname']

        # text input
        input_ids = example['input_ids']
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        text_attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        if self.img_db:
            img_feat, num_bb = self._get_img_feat(img_fname, self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
        else:
            img_feat, img_attn_masks = None, None

        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(img_fname)
            # for the output of speech encoder
            num_segment = int(num_frame/(0.02 * 16000) - 1)
            speech_attn_masks = torch.ones(num_segment, dtype=torch.long)
        else:
            speech_feat = None

        return input_ids, img_feat, speech_feat, text_attn_masks, img_attn_masks, speech_attn_masks, target


def itm_collate(inputs, add_cls_token=True):
    (input_ids, img_feats, speech_feats, text_attn_masks, img_attn_masks, speech_attn_masks, targets
     ) = map(list, unzip(inputs))
    targets = torch.cat(targets, dim=0)
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)
    
    # for gather index
    out_size = text_attn_masks.size(1)

    if img_feats[0] is not None:
        img_attn_masks = pad_sequence(img_attn_masks, batch_first=True, padding_value=0)
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)
        out_size += img_attn_masks.size(1)
    else:
        num_bbs, img_position_ids, img_feat, img_attn_masks = None, None, None, None

    if speech_feats[0] is not None:
        # conv1d downsample 8 times, --- Pending.
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames) # (n, max_num_nbb, dim)
        # for speech encoder output. 关于speech的mask, 采用向下取整的策略, drop last
        speech_attn_masks = pad_sequence(speech_attn_masks, batch_first=True, padding_value=0)
        num_segments = [int(num_frame/(0.02 * 16000) -1) for num_frame in num_frames]
        out_size += speech_attn_masks.size(1)
        speech_position_ids = torch.arange(0, speech_attn_masks.size(1), dtype=torch.long).unsqueeze(0)
        # 这里需要注意的是，speech-feat 经过 speech encoder的长度可能大于speech_attn_masks，
        # 需要在模型部分做一个截断的后处理.
    else:
        num_segments, speech_position_ids, speech_feat, speech_attn_masks = None, None, None, None
    
    bs, max_tl = input_ids.size()
    # multi-modality atten mask
    gather_index = get_gather_index(txt_lens, num_bbs, num_segments, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'txt_lens': txt_lens,
             'text_attn_masks': text_attn_masks,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'num_bbs': num_bbs,
             'img_attn_masks': img_attn_masks,
             'speech_feat': speech_feat,
             'num_segments': num_segments,
             'speech_attn_masks': speech_attn_masks,
             'speech_position_ids': speech_position_ids,
             'gather_index': gather_index,
             'targets': targets}
    return batch