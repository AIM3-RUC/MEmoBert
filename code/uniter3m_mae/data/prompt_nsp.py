"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Prompt NSP dataset
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from code.uniter3m_mae.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
# from uniter 
from code.uniter.data.itm import TokenBucketSamplerForItm, sample_negative

class PromptNSPDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    example['target'] = target
    example['ground_truth_emo'] = value['label']
    """
    def __init__(self, txt_db, img_db, speech_db, neg_sample_p=0.5):
        super().__init__(txt_db, img_db, speech_db)
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db is not None:
            print('[Debug] Img db is not None!!!')
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db is not None:
            print('[Debug] Speech db is not None!!!')
            assert isinstance(speech_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.img_shape = None

    def __getitem__(self, i):
        # i only the index not the real text-id in txtdb
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        target = example['target'] # 0 or 1
        fake_label = example['fake_label'] # 0 or 1
        ground_truth_emo = example['ground_truth_emo'] # {0 1 2 3}
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]    
        # add cls and sep special tokens
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        # must use this, very shit bug that upset me two days
        img_fname = example['img_fname']
        if self.img_db is not None:
            img_feat, num_bb = self._get_img_feat(img_fname, self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, img_attn_masks))
        else:
            img_feat = None

        if self.speech_db is not None:
            speech_feat, num_frame = self._get_speech_feat(img_fname)
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat = None
        target = torch.tensor(target, dtype=torch.long) 
        fake_label = torch.tensor(fake_label, dtype=torch.long) 
        gt_target = torch.tensor(ground_truth_emo, dtype=torch.long)
        return input_ids, img_feat, speech_feat, attn_masks, target, gt_target, fake_label, img_fname


def prompt_nsp_collate(inputs):
    (input_ids, img_feats, speech_feats, attn_masks, targets, gt_targets, fake_labels, img_fnames) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    if img_feats[0] is not None:
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                    ).unsqueeze(0)
    else:
        img_feat, num_bbs, img_position_ids = None, None, None

    if speech_feats[0] is not None:
        ## speech batches
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames)
        # print('[Debug] the batch input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
        speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0)    
    else:
        speech_feat, num_frames, speech_position_ids = None, None, None

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)

    targets = torch.tensor(targets, dtype=torch.long)
    fake_labels = torch.tensor(fake_labels, dtype=torch.long)
    gt_targets = torch.tensor(gt_targets, dtype=torch.long)
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
             'targets': targets,
             'gt_targets': gt_targets,
             'fake_labels': fake_labels,
             'img_fnames': img_fnames}
    return batch