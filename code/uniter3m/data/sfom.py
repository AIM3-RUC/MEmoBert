"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

SFOM, Speech Frame Order Modeling, ref vform.py
"""

import random
from numpy.core.fromnumeric import size
import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
from code.uniter3m.data.vfom import random_reorder

def _get_speech_tgt(target, txt_len, img_len):
    z = torch.ones(txt_len, dtype=torch.long) * -1
    if img_len > 0:
        zi = torch.ones(img_len, dtype=torch.long) * -1
        speech_tgt = torch.cat([z, zi, target], dim=0)
    else:
        speech_tgt = torch.cat([z, target], dim=0)
    return speech_tgt

class SFOMDataset(DetectFeatTxtTokDataset):    
    def __init__(self, random_reorder_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        random_reorder_p:  default=0.15
        '''
        self.random_reorder_p = random_reorder_p
        self.img_shape = None

    def __getitem__(self, i):
        '''
        :add_cls_token, add cls token or not
        '''
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

        if self.img_db:
            img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, img_attn_masks, speech_attn_masks))
        else:
            img_feat, num_bb = None, 0
            attn_masks = torch.cat((attn_masks, speech_attn_masks))            
        
        speech_position_ids = torch.arange(0, speech_feat.size(0), dtype=torch.long)
        # Random shuffle 15% of pos_ids
        speech_orders, speech_order_targets = random_reorder(speech_position_ids, self.random_reorder_p)
        # print('[Debug] speech_orders {}'.format(speech_orders))
        # print('[Debug] speech_order_targets {}'.format(speech_order_targets))
        speech_orders = torch.tensor(speech_orders, dtype=torch.long)
        speech_order_targets = torch.tensor(speech_order_targets, dtype=torch.long)
        # 按真实长度进行填充
        speech_order_targets = _get_speech_tgt(speech_order_targets, len(input_ids), num_bb)
        return (input_ids, img_feat, speech_feat, attn_masks, speech_orders, speech_order_targets)

def sfom_collate(inputs):
    (input_ids, img_feats, speech_feats, attn_masks, speech_orders, speech_order_targets) = map(list, unzip(inputs))
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    
    # speech batches
    num_frames = [f.size(0) for f in speech_feats]
    speech_feat = pad_tensors(speech_feats, num_frames)
    speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)

    if img_feats[0] is not None:
        ## images batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        img_position_ids = torch.arange(0, max(num_bbs), dtype=torch.long).unsqueeze(0)    
    else:
        img_feat, num_bbs, img_position_ids = None, None, None

    all_orders = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0).repeat(speech_feat.size(0), 1)
    for i, nframe in enumerate(num_frames):
        all_orders[i, :nframe] = speech_orders[i]
    # target 仿照mlm的实现方式
    all_targets = pad_sequence(speech_order_targets, batch_first=True, padding_value=-1)

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
             'shuffled_orders': all_orders,
             'targets': all_targets}
    return batch

if __name__ == '__main__':
    # export PYTHONPATH=/data7/MEmoBert
    pos_ids = [0,1,2,3,4,5,6,7,8]
    random_reorder(pos_ids, random_reorder_p=0.3)