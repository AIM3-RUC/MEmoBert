"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MELM datasets: masked emotional language modeling

Update 2020-02-04: Jinming add emo_type_ids
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3flow.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors)
from code.uniter3m.data.melm import random_emo_word
from code.uniter3m.data.data import get_gather_index

class MelmDataset(DetectFeatTxtTokDataset):
    '''
    emotional words masked modeling, the 
    melm_prob: masked probility
    '''
    def __init__(self, mask_prob, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.melm_prob = mask_prob

    def __getitem__(self, i):
        """
        for 
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - text_attn_masks   : (L, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_attn_masks   : (L, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_melm_io(example['input_ids'], example['emo_input_ids'])
        text_attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        # 0 is the no-emotion-words
        if example.get('emo_type_ids') is not None:
            emo_type_ids = torch.tensor([0] + example['emo_type_ids'] + [0])
            # generate the labels for multitask emotion, 保持跟 txt_labels 一致，txt_labels 中为 -1 的位置同样置为-1.
            txt_emo_labels = torch.where(txt_labels<0, txt_labels, emo_type_ids)
            # print("[Debug] txt_labels {}".format(txt_labels))
            # print("[Debug] emo_type_ids {}".format(emo_type_ids))
            # print("[Debug] txt_emo_labels {}".format(txt_emo_labels))
        else:
            txt_emo_labels = None

        if self.img_db:
            # print(f'[Debug] item {i} img is not None')
            img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            self.img_shape = img_feat.shape[1:]
        else:
            # print(f'[Debug] item img {i} is None')
            img_feat = None
        
        if self.speech_db:
            # print(f'[Debug] item {i} speech is not None')
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            # for the output of speech encoder 
            num_segment = int(num_frame/(0.02 * 16000) - 1)
            speech_attn_masks = torch.ones(num_segment, dtype=torch.long)
        else:
            speech_feat = None
        return input_ids, img_feat, speech_feat, text_attn_masks, img_attn_masks, speech_attn_masks, txt_labels, txt_emo_labels

    def create_melm_io(self, input_ids, emo_input_ids):
        input_ids, txt_labels = random_emo_word(self.melm_prob, input_ids, 
                                                    self.txt_db.v_range, emo_input_ids, self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def melm_collate(inputs):
    """
    Jinming: modify to img_position_ids
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_position_ids (n, max_num_bb)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :emo_type_ids (n, max_L) padded with 0
    :txt_emo_labels (n, max_L) padded with -1, similar with the emo_type_ids
    """
    (input_ids, img_feats, speech_feats, text_attn_masks, img_attn_masks, speech_attn_masks, txt_labels, \
             batch_txt_emo_labels) = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)

    # for gather index
    out_size = text_attn_masks.size(1)

    # Jinming: here emo_type_ids is batch, so judge the element is none or not 
    # batch_emo_type_ids is also can used for 
    if batch_txt_emo_labels[0] is not None:
        batch_txt_emo_labels = pad_sequence(batch_txt_emo_labels, batch_first=True, padding_value=-1)
    else:
        batch_txt_emo_labels = None
    
    if img_feats[0] is not None:
        ## image batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        # print('[Debug] batch padding img input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
        img_position_ids = torch.arange(0, max(num_bbs), dtype=torch.long).unsqueeze(0)      
        img_attn_masks = pad_sequence(img_attn_masks, batch_first=True, padding_value=0)
        out_size += img_attn_masks.size(1)
    else:
        img_feat, num_bbs, img_position_ids, img_attn_masks = None, None, None, None
    
    if speech_feats[0] is not None:
        # raw wav and attn degign in model
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames) # (n, max_num_nbb, dim)
        # 关于speech的mask, 采用向下取整的策略, drop last
        speech_attn_masks = pad_sequence(speech_attn_masks, batch_first=True, padding_value=0)
        num_segments = [int(num_frame/(0.02 * 16000) - 1) for num_frame in num_frames]
        out_size += speech_attn_masks.size(1)
        speech_position_ids = torch.arange(0, speech_attn_masks.size(1), dtype=torch.long).unsqueeze(0)
        # 这里需要注意的是，speech-feat 经过 speech encoder的长度可能大于speech_attn_masks，
        # 需要在模型部分做一个截断的后处理.
    else:
        num_segments, speech_position_ids, speech_feat, speech_attn_masks = None, None, None

    bs, max_tl = input_ids.size()
    # multi-modality atten mask
    gather_index = get_gather_index(txt_lens, num_bbs, num_segments, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'txt_lens': txt_lens,
             'text_attn_masks': text_attn_masks,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'num_bbs': num_bbs,
             'img_attn_masks': img_attn_masks,
             'speech_feat': speech_feat,
             'num_segments': num_segments,
             'speech_attn_masks': speech_attn_masks,
             'speech_position_ids': speech_position_ids,
             'gather_index': gather_index,
             'txt_labels': txt_labels, 
             'txt_emo_labels': batch_txt_emo_labels
             }
    return batch
