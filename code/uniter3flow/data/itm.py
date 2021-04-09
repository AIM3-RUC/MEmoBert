"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
from collections import defaultdict
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from code.uniter3flow.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_ids_and_lens)
from code.uniter3flow.data.sampler import TokenBucketSampler


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
    (for more efficient negative sampling) """
    def __init__(self, txt_db, img_db, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
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
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_fname = self.train_imgs[i]
        # img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)
        # Jinming remove norm-bbx feature
        img_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        text_attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        img_attn_masks = torch.ones(num_bb, dtype=torch.long)

        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return input_ids, img_feat, text_attn_masks, img_attn_masks, target


def itm_collate(inputs, add_cls_token=True):
    (input_ids, img_feats, text_attn_masks, img_attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    targets = torch.cat(targets, dim=0)

    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)
    img_attn_masks = pad_sequence(img_attn_masks, batch_first=True, padding_value=0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)

    # add cls token to img branch
    if add_cls_token:
        cls_token_attn_masks = torch.ones((img_attn_masks.size(0), 1), dtype=img_attn_masks.dtype)
        # for img modality 
        img_attn_masks = torch.cat((cls_token_attn_masks, img_attn_masks), dim=1)
        img_position_ids = torch.arange(0, img_feat.size(1)+1, dtype=torch.long).unsqueeze(0)
        # for speech modality -- pending
    else:
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)

    batch = {'input_ids': input_ids,
             'txt_lens': txt_lens,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'num_bbs': num_bbs,
             'img_position_ids': img_position_ids,
             'text_attn_masks': text_attn_masks,
             'img_attn_masks': img_attn_masks,
             'targets': targets}

    return batch


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad