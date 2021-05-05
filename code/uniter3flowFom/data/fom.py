"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

FOM datasets
refer:
https://github.com/linjieli222/HERO/blob/faaf15d6ccc3aa4accd24643d77d75699e9d7fae/data/fom.py
"""
import random
from numpy.core.fromnumeric import size

import copy
import torch
from torch._C import dtype
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3flowFom.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb, pad_tensors)

'''
For Image Contextual Representations Or Speech Contextual Representations.
FOM 是 Frame Order Moding, 这里主要是最 Image 做. 如果有Speech信息那么全部保留.

'''
class FOMDataset(DetectFeatTxtTokDataset):
    def __init__(self, random_reorder_p, txt_db, img_db, speech_db):
        '''
        random_reorder_p:  default=0.15
        '''
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db:
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db:
            assert isinstance(speech_db, DetectFeatLmdb)
        self.random_reorder_p = random_reorder_p
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.img_shape = None

    def __getitem__(self, i):
        '''
        :add_cls_token, add cls token or not
        '''
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        text_attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        if self.img_db:
            img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
        else:
            img_feat, img_attn_masks = None, None
        
        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            # for the output of speech encoder 
            num_segment = int(num_frame/(0.02 * 16000) - 1)
            speech_attn_masks = torch.ones(num_segment, dtype=torch.long)
        else:
            speech_feat, speech_attn_masks = None, None        
        
        img_position_ids = torch.arange(0, img_feat.size(0), dtype=torch.long).unsqueeze(0)
        # Random shuffle 15% of pos_ids
        img_orders, img_targets = random_reorder(
            list(range(len(img_position_ids))), self.random_reorder_p)
        print('[Debug] img_orders {}'.format(img_orders))
        print('[Debug] img_targets {}'.format(img_targets))
        img_orders = torch.tensor(img_orders, dtype=torch.long)
        img_order_targets = torch.tensor(img_targets, dtype=torch.long)
        return input_ids, img_feat, speech_feat, text_attn_masks, img_attn_masks, img_orders, img_order_targets

def fom_collate(inputs):
    (input_ids, img_feats, speech_feats, text_attn_masks, img_attn_masks, img_orders, img_order_targets
    ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)

    num_bbs = [item.size(0) for item in img_feats]
    all_orders = torch.arange(
        0, img_feats.size(1), dtype=torch.long).unsqueeze(0).repeat(img_feats.size(0), 1)
    img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)
    all_targets = torch.ones_like(all_orders) * -1
    for i, nframe in enumerate(num_bbs):
        all_orders[i, :nframe] = img_orders[i]
        all_targets[i, :nframe] = img_order_targets[i]
    # 计算
    reordered_frame_idx = []
    binary_targets = []
    bs, max_bb = all_orders.size()
    for clip_idx in range(bs):
        for i in range(num_bbs[clip_idx]):
            # 每个video的每一帧,
            if all_targets[clip_idx, i] == -1:
                # 如果不是有效帧，Pass
                continue
            for j in range(i+1, num_bbs[clip_idx]):
                # 对于有效的第 i 帧, 在i的后面后面找一个有效的 j 帧
                if all_targets[clip_idx, j] == -1:
                    continue
                reordered_frame_idx.append(clip_idx*max_bb+i)
                reordered_frame_idx.append(clip_idx*max_bb+j)
                # 如果 j 小于 i, 说明该帧是 shuffle, 即 binary=0
                if all_targets[clip_idx, i] > all_targets[clip_idx, j]:
                    binary_targets.append(0)
                else:
                    binary_targets.append(1)
                reordered_frame_idx.append(clip_idx*max_bb+j)
                reordered_frame_idx.append(clip_idx*max_bb+i)
                if all_targets[clip_idx, j] > all_targets[clip_idx, i]:
                    binary_targets.append(0)
                else:
                    binary_targets.append(1)
    reordered_frame_idx = torch.tensor(reordered_frame_idx, dtype=torch.long)
    binary_targets = torch.tensor(binary_targets, dtype=torch.long)
    print('[Debug] all_orders {}'.format(all_orders[0]))
    print('[Debug] all_targets {}'.format(all_targets[0]))
    print('[Debug] reordered_frame_idx {}'.format(reordered_frame_idx[0]))
    print('[Debug] binary_targets {}'.format(binary_targets[0]))

    if speech_feats:
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames) # (n, max_num_nbb, dim)
    else:
        num_frames, speech_feat = None, None

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'txt_lens': txt_lens,
            'text_attn_masks': text_attn_masks,
             'img_feat': img_feat,
             'num_bbs': num_bbs,
            'img_attn_masks': img_attn_masks,
             'speech_feat': speech_feat,
             'num_frames': num_frames,
             'shuffled_orders': all_orders,
             'targets': all_targets,
             'reordered_frame_idx': reordered_frame_idx,
             'binary_targets': binary_targets }
    return batch


def random_reorder(pos_ids, random_reorder_p=0.15):
    """
    random reorder frame positions
    每帧以15%的概率选中进行，然后以预测 整个序列原来的顺序。
    return:
    for example: select pos [0, 2, 3, 4]
                   shuffle: [2, 3, 0, 4]
    将shuffle后的位置放到原来的序列中, 然后其中1和5不会变.
    output_order: [2, 1, 3, 0, 4, 5]
    原始2对应shuffle后的3, 原始3对应shuffle后的0, ...
    output_target: [3, -1, 0, 2, 4, -1]
    """
    selected_pos = []
    target_pos = []
    # step1: 选择哪些位置的id将会被打乱
    for i, pos_id in enumerate(pos_ids):    
        prob = random.random()
        # mask token with 15% probability
        if prob < random_reorder_p:
            selected_pos.append(i)
            target_pos.append(pos_id)
    print('[Debug] select pos {}'.format(selected_pos))
    print('[Debug] target pos {}'.format(target_pos))
    # step2: 将选中的目标进行打乱
    target_pos_shuffled = copy.deepcopy(target_pos)
    random.shuffle(target_pos_shuffled)
    print('[Debug] target_pos_shuffled {}'.format(target_pos_shuffled))
    output_order = copy.deepcopy(pos_ids)
    output_target = [-1] * len(output_order)
    for i, pos in enumerate(selected_pos):
        output_order[pos] = target_pos_shuffled[i]
        output_target[target_pos_shuffled[i]] = pos
    print('[Debug] output_order {}'.format(output_order))
    print('[Debug] output_target {}'.format(output_target))
    return output_order, output_target

if __name__ == '__main__':
    # export PYTHONPATH=/data7/MEmoBert
    pos_ids = [0,1,2,3,4,5,6,7,8]
    random_reorder(pos_ids, random_reorder_p=0.3)