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
from code.uniter3flow_speech.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb, pad_tensors)

class FomDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, speech_db, random_reorder_p=0.15):
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db:
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db:
            assert isinstance(speech_db, DetectFeatLmdb)
        self.random_reorder_p = random_reorder_p
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid_ = self.ids[i]
        (f_sub_input_ids, f_v_feats, f_attn_masks,
         c_v_feats, c_attn_masks,
         num_subs, sub2frames) = self.vid_sub_db[vid_]
        c_pos_ids = [i for i in range(len(c_v_feats))]
        # Random shuffle 15% of pos_ids
        orders, targets = random_reorder(
            list(range(len(c_pos_ids))), self.random_reorder_p)
        orders = torch.tensor(orders, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        video_inputs = (
            f_sub_input_ids, f_v_feats, f_attn_masks,
            c_v_feats, c_attn_masks,
            num_subs, sub2frames)
        out = (video_inputs, orders, targets)
        return out


def fom_collate(inputs):
    (video_inputs, orders, targets) = map(list, unzip(inputs))
    batch = video_collate(video_inputs)

    clip_level_v_feats = batch["c_v_feats"]
    num_frames = [item.size(0) for item in orders]

    all_orders = torch.arange(
        0, clip_level_v_feats.size(1), dtype=torch.long).unsqueeze(0).repeat(
            clip_level_v_feats.size(0), 1)
    all_targets = torch.ones_like(all_orders) * -1
    for i, nframe in enumerate(num_frames):
        all_orders[i, :nframe] = orders[i]
        all_targets[i, :nframe] = targets[i]
    reordered_frame_idx = []
    binary_targets = []
    bs, max_vl = all_orders.size()
    for clip_idx in range(bs):
        for i in range(num_frames[clip_idx]):
            if all_targets[clip_idx, i] == -1:
                continue
            for j in range(i+1, num_frames[clip_idx]):
                if all_targets[clip_idx, j] == -1:
                    continue
                reordered_frame_idx.append(clip_idx*max_vl+i)
                reordered_frame_idx.append(clip_idx*max_vl+j)
                if all_targets[clip_idx, i] > all_targets[clip_idx, j]:
                    binary_targets.append(0)
                else:
                    binary_targets.append(1)

                reordered_frame_idx.append(clip_idx*max_vl+j)
                reordered_frame_idx.append(clip_idx*max_vl+i)
                if all_targets[clip_idx, j] > all_targets[clip_idx, i]:
                    binary_targets.append(0)
                else:
                    binary_targets.append(1)
    reordered_frame_idx = torch.tensor(reordered_frame_idx, dtype=torch.long)
    binary_targets = torch.tensor(binary_targets, dtype=torch.long)
    batch["shuffled_orders"] = all_orders
    batch["targets"] = all_targets
    batch['reordered_frame_idx'] = reordered_frame_idx
    batch['binary_targets'] = binary_targets
    return batch

def random_reorder(pos_ids, random_reorder_p=0.15):
    """
    random reorder frame positions
    """
    selected_pos = []
    target_pos = []
    for i, pos_id in enumerate(pos_ids):
        prob = random.random()
        # mask token with 15% probability
        if prob < random_reorder_p:
            selected_pos.append(i)
            target_pos.append(pos_id)
    target_pos_shuffled = copy.deepcopy(target_pos)
    random.shuffle(target_pos_shuffled)
    output_order = copy.deepcopy(pos_ids)
    output_target = [-1] * len(output_order)
    for i, pos in enumerate(selected_pos):
        output_order[pos] = target_pos_shuffled[i]
        output_target[target_pos_shuffled[i]] = pos
    return output_order, output_target