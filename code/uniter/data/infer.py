"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Evaluation dataset
顺序的读取数据的Batch
"""
 
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
import numpy as np

from code.uniter.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index)

class InferDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, batch_size):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        super().__init__(txt_db, img_db)
        self.bs = batch_size

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids = example['input_ids']
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])

        # img input Jinming remove the norm-bbx fts
        img_feat, num_bb = self._get_img_feat(example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, attn_masks

def infer_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len], with cls and sep 
    :img_feat     (n, max_num_bb, feat_dim)
    :img_position_ids (n, max_num_bb)
    :num_bbs      list of [num_bb], real num_bbs
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    """
    (input_ids, img_feats, attn_masks) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)
    img_position_ids = torch.arange(0, img_feat.size(0), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'txt_lens': txt_lens,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'img_lens': num_bbs,
             'attn_masks': attn_masks,
             'gather_index': gather_index}
    return batch