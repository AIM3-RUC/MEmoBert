"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MSRM Datasets, mask speech regression modeling
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
from code.uniter3m.data.mrm import _get_img_mask, _get_feat_target, _mask_img_feat


def _get_speech_tgt_mask(speech_mask, txt_len, img_len):
    # 如果三个模态都有，那么保持text, img, speech 的组合顺序
    z = torch.zeros(txt_len, dtype=torch.bool)
    if img_len > 0:
        zi = torch.zeros(img_len, dtype=torch.bool)
        speech_mask_tgt = torch.cat([z, zi, speech_mask], dim=0)
    else:
        speech_mask_tgt = torch.cat([z, speech_mask], dim=0)
    return speech_mask_tgt

class MsrfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.mask_prob = mask_prob
        self.img_shape = None


    def __getitem__(self, i):
        """
        保持 text, img, speech 的组合顺序
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - speech_feat     : (num_bb, d)
        - speech_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - speech_mask     : (num_bb, ) between {0, 1}
        """
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        
        # speech input
        speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
        speech_attn_masks = torch.ones(num_frame, dtype=torch.long)

        if not self.img_db:
            img_feat, num_bb = None, 0
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, img_attn_masks, speech_attn_masks))
           
        speech_mask = _get_img_mask(self.mask_prob, num_frame)
        speech_mask_tgt = _get_speech_tgt_mask(speech_mask, len(input_ids), num_bb)
        return (input_ids, img_feat, speech_feat, attn_masks, speech_mask, speech_mask_tgt)


def msrfr_collate(inputs):
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
    (input_ids, img_feats, speech_feats, attn_masks, speech_masks, speech_mask_tgts,
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    # speech info
    num_frames = [f.size(0) for f in speech_feats]
    speech_feat = pad_tensors(speech_feats, num_frames)
    speech_position_ids = torch.arange(0, max(num_frames), dtype=torch.long
                                ).unsqueeze(0)

    # mask features
    speech_masks = pad_sequence(speech_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(speech_feat, speech_masks)
    speech_feat = _mask_img_feat(speech_feat, speech_masks)
    speech_mask_tgt = pad_sequence(speech_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    if img_feats[0] is not None:
        ## images batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        img_position_ids = torch.arange(0, max(num_bbs), dtype=torch.long).unsqueeze(0)    
    else:
        img_feat, num_bbs, img_position_ids = None, None, None

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
             'speech_masks': speech_masks,
             'speech_mask_tgt': speech_mask_tgt}
    return batch