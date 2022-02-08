"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

FOM, Frame Order Modeling for visual
refer:
https://github.com/linjieli222/HERO/blob/faaf15d6ccc3aa4accd24643d77d75699e9d7fae/data/fom.py
https://github.com/linjieli222/HERO/blob/7be5e039361ef2afbc2ce1323dcb1ad927034f5b/model/model.py#L306

任务是什么？
输入是原始的顺序的feature序列，经过网络得到输出。
shuffle_output_order: [4, 7, 2, 3, 8, 5, 6, 0, 1]
orderindex=4的部分真实的序号是0, orderindex=7的真实的1, orderindex=8的真实位置是4

output_target: [7, 8, -1, -1, 0, -1, -1, 1, 4]
然后经过 scatter 函数将输入特征按照 shuffle_output_order 进行乱序，然后经过网络之后预测当前位置特征的真实的顺序。
"""
import random
from numpy.core.fromnumeric import size

import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m_mae.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index

def random_reorder(pos_ids, random_reorder_p=0.15):
    """
    random reorder frame positions
    每帧以15%的概率选中进行，然后以预测 shuffle的部分原来的序号。
    Note: 至少shuffle一个吧否则会报错。
    return:
    for example: select pos: [0, 1, 4, 7, 8]
                 target pos: [0, 1, 4, 7, 8]
         shuffle target pos: [4, 7, 8, 0, 1]
   将shuffle后的位置放到原来的序列中, 然后其中23和56不会变.
   output_order = shuffled_order: [4, 7, 2, 3, 8, 5, 6, 0, 1]
    按照shuffle order将输入的特征序列按照以上的顺序重新组合，所以此时0号位置应该是放time=4的特征，1号位置是time=7的特征
   那 output_target 和 output_order 的对应关系是什么？
   time=0的特征在7号位置上，time=1的特征在8号位置上。 预测真实的时刻的特征所在的位置。
   output_target: [7, 8, -1, -1, 0, -1, -1, 1, 4]
    """
    selected_pos = []
    target_pos = []
    # step1: 选择哪些位置的 id 将会被打乱
    count = 0
    for i, pos_id in enumerate(pos_ids):    
        prob = random.random()
        # mask token with 15% probability
        if prob < random_reorder_p:
            selected_pos.append(i)
            target_pos.append(pos_id)
            count += 1
    if count == 0:
        # at least mask 1
        rand_pos = random.randint(0, len(pos_ids)-1)
        selected_pos.append(rand_pos)
        target_pos.append(rand_pos)       
    # print('[Debug] select pos {}'.format(selected_pos))
    # print('[Debug] target pos {}'.format(target_pos))
    # step2: 将选中的目标进行打乱
    target_pos_shuffled = copy.deepcopy(target_pos)
    random.shuffle(target_pos_shuffled)
    # print('[Debug] target_pos_shuffled {}'.format(target_pos_shuffled))
    output_order = copy.deepcopy(pos_ids)
    output_target = [-1] * len(output_order)
    for i, pos in enumerate(selected_pos):
        output_order[pos] = target_pos_shuffled[i]
        output_target[target_pos_shuffled[i]] = pos
    # print('[Debug] output_order {}'.format(output_order))
    # print('[Debug] output_target {}'.format(output_target))
    return output_order, output_target

def _get_img_tgt(target, txt_len, speech_len):
    z = torch.ones(txt_len, dtype=torch.long) * -1
    if speech_len > 0:
        zs = torch.ones(speech_len, dtype=torch.long) * -1
        img_tgt = torch.cat([z, target, zs], dim=0)
    else:
        img_tgt = torch.cat([z, target], dim=0)
    return img_tgt

class VFOMDataset(DetectFeatTxtTokDataset):    
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

        # img input
        img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
        self.img_shape = img_feat.shape[1:]
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat, num_frame = None, 0
        
        img_position_ids = torch.arange(0, img_feat.size(0), dtype=torch.long)
        # Random shuffle 15% of pos_ids
        img_orders, img_order_targets = random_reorder(img_position_ids, self.random_reorder_p)
        # print('[Debug] img_orders {}'.format(img_orders))
        # print('[Debug] img_order_targets {}'.format(img_order_targets))
        # 按真实长度进行填充
        img_order_targets = torch.tensor(img_order_targets, dtype=torch.long)
        img_order_targets = _get_img_tgt(img_order_targets, len(input_ids), num_frame)
        return (input_ids, img_feat, speech_feat, attn_masks, img_orders, img_order_targets)

def vfom_collate(inputs):
    (input_ids, img_feats, speech_feats, attn_masks, img_orders, img_order_targets) = map(list, unzip(inputs))
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    
    # img batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)

    if speech_feats[0] is not None:
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames) # (n, max_num_nbb, dim)
        speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0)    
    else:
        num_frames, speech_feat, speech_position_ids = None, None, None

    all_orders = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0).repeat(img_feat.size(0), 1)
    for i, nframe in enumerate(num_bbs):
        all_orders[i, :nframe] = img_orders[i]
    # target 仿照mlm的实现方式
    all_targets = pad_sequence(img_order_targets, batch_first=True, padding_value=-1)

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