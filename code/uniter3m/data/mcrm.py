"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Mask Consecutive Region Modeling Datasets
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
# from uniter 
from code.uniter.data.mrm import _get_img_mask, _get_feat_target, _mask_img_feat, _get_targets

def _get_consecutive_img_mask(mask_prob, num_bb, mask_consecutive=3):
    # 在中间部分，连续mask3帧左右的数据, 同时保持跟BERT一致，同时生成 Target.
    # 一个句子只mask一个span.
    # https://github.com/andi611/Mockingjay-Speech-Representation/blob/9377bf2585c020b4d217b35f0d27963eb45274ef/utility/mam.py#L92
    img_mask = [0 for _ in range(num_bb)]
    # determine whether to mask / random / or do nothing to the frame
    dice = random.random()
    valid_index_range = int(num_bb - mask_consecutive - 1) # compute valid len for consecutive masking
    proportion = int(num_bb * 0.15 // mask_consecutive)
    if proportion == 0:
        # 不满足 mask span 条件
        img_mask[random.choice(range(num_bb))] = True
    else:
        chosen_index = torch.randperm(valid_index_range)[:proportion] # draw `proportion` samples from the range (0, valid_index_range) and without replacement
        # mask to zero
        if bool(dice < 0.8):
            for i in range(mask_consecutive):
                img_mask[chosen_index+i] = True
        # replace to random frames
        elif bool(dice >= 0.8) and bool(dice < 0.9):
            random_index = torch.randperm(valid_index_range).data.cpu().numpy()[:proportion]
            for i in range(mask_consecutive):
                img_mask[chosen_index+i] = img_mask[random_index+i]
        # do nothing
        else:
            pass

    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
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

class MrfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.mask_prob = mask_prob
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

        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids), num_frame)
        return (input_ids, img_feat, speech_feat, attn_masks, img_mask, img_mask_tgt)


def mrfr_collate(inputs):
    """
    Return:
    - input_ids    : (n, max_L), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
    - position_ids : (n, max_L)
    - txt_lens     : list of [input_len]
    - img_feat     : (n, max_num_bb, d)
    - img_position_ids : (n, max_num_bb)
    - num_bbs      : list of [num_bb]
    - attn_masks   : (n, max_{L + num_bb}), ie., [1, 1, ..., 0, 0, 1, 1]
    - img_masks    : (n, max_num_bb) between {0, 1}
    """
    (input_ids, img_feats, speech_feats, attn_masks, img_masks, img_mask_tgts,
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

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
    # set number-frames to None
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt}
    return batch

class MrcDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.mask_prob = mask_prob
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
        img_mask = _get_img_mask(self.mask_prob, num_bb)

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


def mrc_collate(inputs):
    (input_ids, img_feats, speech_feats, img_soft_labels,
     attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    img_feat = pad_tensors(img_feats, num_bbs)
    img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)
    img_soft_label = pad_tensors(img_soft_labels, num_bbs)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    label_targets = _get_targets(img_masks, img_soft_label)

    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)
        
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

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
    # set number-frames to None
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'label_targets': label_targets}
    return batch