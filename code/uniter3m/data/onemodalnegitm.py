"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
OneModalNegItmDataset dataset, 只替换其中的一个模态。
Only for the speech and img both exists
Case1: 随机替换其中的语音模态
Case2: 随机替换其中的视觉模态
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from code.uniter3m.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
# from uniter 
from code.uniter.data.itm import TokenBucketSamplerForItm, sample_negative

class OneModalNegItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) 
    Only for the speech and img both exists
    """
    def __init__(self, txt_db, img_db, speech_db, neg_sample_p=0.8, neg_img_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        assert isinstance(speech_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.img_shape = None

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))

        self.neg_sample_p = neg_sample_p
        self.neg_img_p = neg_img_p
        print(f'[Info OneModalNegItmDataset] neg_sample_p {neg_sample_p} and neg_img_p {neg_img_p}')
        self.new_epoch()
    
    def new_epoch(self):
        """ should be called every epoch for more randomness
        every epoch to random this
        首先随机生成0or1的labels, 如果label=0, 那么就根据文本构建一个image的负例 或者 语音的负例。
        """
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1-self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        self.train_speechs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            pos_img_fname = super().__getitem__(i)['img_fname']
            if self.labels[i] == 0:
                neg_img_fname = sample_negative(self.all_imgs, [pos_img_fname], 1)[0]
                prob = random.random()
                if prob > self.neg_img_p:
                    # 如果概率大于 neg_img_p 那么替换 img 模态作为负例
                    self.train_imgs.append(neg_img_fname)
                    self.train_speechs.append(pos_img_fname)
                    self.lens.append(tl + self.img_db.name2nbb[neg_img_fname] + self.speech_db.name2nbb[pos_img_fname])
                else:
                    # 如果概率小于 neg_img_p 那么替换 speech 模态作为负例
                    self.train_imgs.append(pos_img_fname)
                    self.train_speechs.append(neg_img_fname)
                    self.lens.append(tl + self.img_db.name2nbb[pos_img_fname] + self.speech_db.name2nbb[neg_img_fname])
            else:
                # 正例
                self.train_imgs.append(pos_img_fname)
                self.train_speechs.append(pos_img_fname)
                self.lens.append(tl + self.img_db.name2nbb[pos_img_fname] + self.speech_db.name2nbb[pos_img_fname])

    def __getitem__(self, i):
        # i only the index not the real text-id in txtdb
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        # text input
        input_ids = example['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]    
        # add cls and sep special tokens
        input_ids = self.txt_db.combine_inputs(input_ids)

        # must use this prebuilded train_imgs, very shit bug that upset me two days
        img_fname = self.train_imgs[i]
        img_feat, num_bb = self._get_img_feat(img_fname, self.img_shape)
        self.img_shape = img_feat.shape[1:]
        # for speech modality 
        speech_fname = self.train_speechs[i]
        speech_feat, num_frame = self._get_speech_feat(speech_fname)

        attn_masks = torch.ones(len(input_ids) + num_bb + num_frame, dtype=torch.long)

        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)
        return input_ids, img_feat, speech_feat, attn_masks, target


def onemodal_negitm_collate(inputs):
    (input_ids, img_feats, speech_feats, attn_masks, targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    ## img batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)

    ## speech batches
    num_frames = [f.size(0) for f in speech_feats]
    speech_feat = pad_tensors(speech_feats, num_frames)
    # print('[Debug] the batch input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
    speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0)    

    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)

    # print(f'[Debug in itm data] text {input_ids.shape} img {img_feat.shape} speech {speech_feat}')
    # print(f'[Debug in itm data] targets {targets.shape} gather_index {gather_index.shape} attn_masks {attn_masks.shape}')

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch