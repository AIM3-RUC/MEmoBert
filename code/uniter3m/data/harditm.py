"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Hard Itm dataset, one-positive and many negative. Not done.
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

class HardItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself"""
    def __init__(self, txt_db, img_db, speech_db, neg_samples=150):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db, speech_db)

        assert isinstance(txt_db, TxtTokLmdb)
        if img_db is not None:
            print('[Debug] Img db is not None!!!')
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db is not None:
            print('[Debug] Speech db is not None!!!')
            assert isinstance(speech_db, DetectFeatLmdb)
        
        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.txt_name_list = list(self.txt2img.keys())
        self.neg_sample_size = neg_sample_size

        self.img_shape = None

    def __getitem__(self, i):
        # Generate one batch, not item.
        # one-positive and many negative samples
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]
        # one-image can also get only on img_id
        gt_txt_ids = self.img2txts[gt_img_id]

        # sample negative
        neg_txt_ids = sample_negative(self.txt_name_list, gt_txt_ids, self.neg_sample_size)
        txt_ids = [gt_txt_id] + neg_txt_ids

        # process text inputs
        all_inputs = []
        txt_lens = []
        for txt_id in txt_ids:
            input_ids = self.txt_db[txt_id]['input_ids']
            if isinstance(input_ids[0], list):
                # for whole word masking 
                input_ids = [y for x in input_ids for y in x]   
            # add cls and sep special tokens 
            input_ids = self.txt_db.combine_inputs(input_ids)
            all_inputs.append(input_ids)
            txt_lens.append(len(input_ids))
       
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

        # process image and speech features (gt always first)
        if self.img_db is not None:
            img_feat, nbb = self._get_img_feat(gt_img_id, self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_feat = img_feat.unsqueeze(0)
            img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)
        else:
            img_feat = None
        if self.speech_db is not None:
            speech_feat, num_frame = self._get_speech_feat(gt_img_id)
        else:
            speech_feat = None

        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_position_ids': img_position_ids,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch

def harditm_collate(inputs):
    # in get_item is buulding one batch data
    assert len(inputs) == 1
    return inputs[0]