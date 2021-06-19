"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MERM Datasets, 根据论文挑选一些重要的部分进行Mask效果对于下游任务可能会好一些。类似与 Mask Emotion Words.
然后添加两个multi-task任务，预测情感帧 以及 情感帧的类别预测。
Train No Evil: Selective Masking for Task-Guided Pre-Training. 2020 EMNLP

Note: 跟MRM的唯一的区别是遮蔽的时候只遮蔽情感帧.
use the collect funtions in mrm.py
"""

import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
# from uniter 
from code.uniter.data.mrm import _get_feat_target, _mask_img_feat, _get_targets

def _get_emo_img_mask(num_bb, soft_labels):
    '''
    soft_labels: (numbbs, 5)
    # emo-list: ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']
    mask 情感显著帧, 如果某帧的情感类别不是neu, 那么以30%的概率遮蔽.
    '''
    img_mask = []
    for index in range(num_bb):
        emo_cate = np.argmax(soft_labels[index])
        if emo_cate > 0:
            prob = random.random()
            if prob < 0.3:
                img_mask.append(True)
            else:
                img_mask.append(False)
        else:
            img_mask.append(False)
    # print(f'[Debug in MERM] img_mask {img_mask}')
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _get_img_tgt_mask(img_mask, txt_len, speech_len):
    if float(torch.__version__[:3]) <= 1.2:
        bool_type = torch.uint8
    else:
        bool_type = torch.bool
    z = torch.zeros(txt_len, dtype=bool_type)
    if speech_len > 0:
        zs = torch.zeros(speech_len, dtype=bool_type)
        img_mask_tgt = torch.cat([z, img_mask, zs], dim=0)
    else:
        img_mask_tgt = torch.cat([z, img_mask], dim=0)
    return img_mask_tgt

class MerfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.img_shape = None

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask     : (num_bb, ) between {0, 1}
        """
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]
        input_ids = self.txt_db.combine_inputs(input_ids)

        img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
        self.img_shape = img_feat.shape[1:]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        
        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat, num_frame = None, 0

        soft_labels = self.img_db.get_dump(example['img_fname'])['soft_labels']
        # print(f'[Debug in merm] soft_labels {soft_labels.shape} nbb {num_bb}')
        img_mask = _get_emo_img_mask(num_bb, soft_labels)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids), num_frame)
        return (input_ids, img_feat, speech_feat, attn_masks, img_mask, img_mask_tgt)

class MercDataset(DetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.img_shape = None

    def _get_img_feat(self, fname, img_shape):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        if num_bb == 0:
            if img_shape is None:
                print("[Warning] Set the img_shape to 342!!!")
                img_shape = 342   
            img_feat = torch.zeros(img_shape).unsqueeze(0)
            img_soft_label = torch.zeros(8).unsqueeze(0)
            # set to neutral
            img_soft_label[0][0] = 1
            num_bb = 1
        return img_feat, img_soft_label, num_bb

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_soft_labels, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
        self.img_shape = img_feat.shape[1:]

        # image input features
        img_mask = _get_emo_img_mask(num_bb, img_soft_labels)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat, num_frame = None, 0
        
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids), speech_len=num_frame)
        return (input_ids, img_feat, speech_feat,
                img_soft_labels, attn_masks, img_mask, img_mask_tgt)