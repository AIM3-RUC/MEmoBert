"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Mask Span Region Modeling Datasets

采用 Mockingjay 的完全类似 Bert MLM 的做法来做 acoustic and visual frames modeling.
# https://github.com/andi611/Mockingjay-Speech-Representation/blob/9377bf2585c020b4d217b35f0d27963eb45274ef/utility/mam.py#L92
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index

def get_consecutive_mask(num_bb, mask_consecutive=3):
    # random replacement 部分的数据 tokens 不是 1000. 所以在构建 img_target_mask 的时候会缺失，因此在构建的时候多保留一个 img_tgt_mask.
    # 或者构建 img_tgt_mask 的时候根据 output_label 来构建。
    MASK = 1000 # MASK 表示该位置被Mask, 方便后面单独处理
    # 一次只Mask一个Span, 要求长度大于10
    tokens = list(range(num_bb))
    output_label = [-1 for _ in range(num_bb)] # mask 之后对应的target帧的标签
    # determine whether to mask / random / or do nothing to the frame
    valid_index_range = int(num_bb - mask_consecutive - 1)
    proportion = int(num_bb * 0.20 // mask_consecutive) # 长度为10即可Mask. 10*0.2/3=1

    if proportion == 0:
        # 不满足 mask-span 条件，随机遮蔽一个
        random_mask = random.randint(0, num_bb-1)
        tokens[random_mask] = MASK
        output_label[random_mask] = random_mask
    else:
        dice = random.random()
        chosen_indexs = random.sample(list(range(valid_index_range)), proportion) # draw `proportion` samples from the range (0, valid_index_range) and without replacement
        # print(f'Debug chosen_indexs are {chosen_indexs}')
        # mask to zero
        if bool(dice < 0.8):
            # print(f'Debug normally are {chosen_indexs}')
            for chosen_index in chosen_indexs:
                for i in range(mask_consecutive):
                    tokens[chosen_index+i] = MASK
                    output_label[chosen_index+i] = chosen_index+i
        # replace to random frames
        elif bool(dice >= 0.8) and bool(dice < 0.9):
            # print(f'Debug random replacement are {chosen_indexs}')
            random_indexs = random.sample(list(range(valid_index_range)), proportion)
            for chosen_index, random_index in zip(chosen_indexs, random_indexs):
                for i in range(mask_consecutive):
                    tokens[chosen_index+i] = tokens[random_index+i]
                    output_label[chosen_index+i] = chosen_index+i
        # do nothing
        else:
            pass
    if sum(output_label) == len(output_label) * -1:
        # at least mask 1
        random_mask = random.randint(0, num_bb-1)
        tokens[random_mask] = MASK
        output_label[random_mask] = random_mask        
    return tokens, output_label

def _get_img_tgt_mask(output_label, txt_len, speech_len):
    '''
    img_frames, img_labels, num_frame
    根据 output_label 来构建img_tgt_mask
    '''
    img_tgt_mask = [False for i in range(len(output_label))]
    for i in range(len(output_label)):
        if output_label[i] != -1:
            img_tgt_mask[i] = True
    if float(torch.__version__[:3]) <= 1.2:
        bool_type = torch.uint8
    else:
        bool_type = torch.bool
    img_tgt_mask = torch.tensor(img_tgt_mask, dtype=bool_type)
    z = torch.zeros(txt_len, dtype=bool_type)
    if speech_len > 0:
        zs = torch.zeros(speech_len, dtype=bool_type)
        img_mask_tgt = torch.cat([z, img_tgt_mask, zs], dim=0)
    else:
        img_mask_tgt = torch.cat([z, img_tgt_mask], dim=0)
    return img_mask_tgt

def _get_feat_target(one_img_feat, output_label):
    # one_img_feat of one video, 保持与img_tgt_mask一致
    assert len(one_img_feat) == len(output_label)
    feat_target = []
    for i in range(len(output_label)):
        if output_label[i] != -1:
            feat_target.append(one_img_feat[i])
    return feat_target

def _get_target(img_soft_label, output_label):
    # 获取真实的 soft-taget
    assert len(img_soft_label) == len(output_label)
    soft_target = []
    for i in range(len(output_label)):
        if output_label[i] != -1:
            soft_target.append(img_soft_label[i])
    return soft_target

def _mask_img_feat(img_feat, img_masks):
    # 将masked的部分置为0向量
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

class MSpanrfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_consecutive, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.mask_consecutive = mask_consecutive
        self.img_shape = None

    def __getitem__(self, i):
        """
        # consider the situation of no text
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask     : (num_bb, ) between {0, 1}
        """
        example = super().__getitem__(i)

        if self.no_text:
            # 只保留cls分类位置.
            # print('[Debug in MSpanrfrDataset] no text info!!!')
            input_ids = [self.txt_db.cls_]
            input_ids = torch.tensor(input_ids)
        else:
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

        # 获取 img_mask_tgt and img_mask
        img_frames, output_label = self.create_mcrm_io(num_bb, mask_consecutive=self.mask_consecutive)
        img_mask = torch.tensor([True if frame == 1000 else False for frame in img_frames], dtype=torch.long)
        # print(f'[Debug in mspanrfr] img mask {img_mask} {len(img_mask)}')
        img_mask_tgt = _get_img_tgt_mask(output_label, len(input_ids), num_frame)
        # 在修改feature之前先获取target.
        # print(f'[Debug in mspanrfr] img feat {img_feat.shape}')
        feat_target = _get_feat_target(img_feat, output_label)
        feat_target = torch.cat(feat_target).reshape(-1, img_feat.size(-1))
        # print(f'[Debug in mspanrfr] masked feat_targets {feat_target.shape}')
        assert sum(img_mask_tgt) == feat_target.size(0)
        # 将random replacement 的帧替换一下
        for i, index in enumerate(img_frames):
            if index != 1000 and i != index:
                img_feat[i] = img_feat[index]
                # print(f'[Debug in mspanrfr] sample {i} replacement feature indexs {i} to {index}')
        return (input_ids, img_feat, speech_feat, attn_masks, img_mask, img_mask_tgt, feat_target)

    def create_mcrm_io(self, num_bb, mask_consecutive=3):
        input_frames, img_labels = get_consecutive_mask(num_bb, mask_consecutive)
        input_frames = torch.tensor(input_frames)
        img_labels = torch.tensor(img_labels)
        return input_frames, img_labels

def mspanrfr_collate(inputs):
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
    (input_ids, img_feats, speech_feats, attn_masks, img_masks, img_mask_tgts, feat_targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    # 不需要pad，只需要将所有masked的target拼接起来即可
    feat_targets = torch.cat(feat_targets, dim=0).view(-1, img_feat.size(-1))
    # print(f'[Debug in mspanrfr] masked batch feat_targets {feat_targets.shape}')
    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)
    assert torch.sum(img_mask_tgt) == feat_targets.size(0)
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

class MSpanrcDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_consecutive, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for visual feature modeling
        '''
        self.mask_consecutive = mask_consecutive
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

        if self.no_text:
            # 只保留cls分类位置.
            # print('[Debug in MSpanrfrDataset] no text info!!!')
            input_ids = [self.txt_db.cls_]
            input_ids = torch.tensor(input_ids)
        else:
            # text input
            input_ids = example['input_ids']
            if isinstance(input_ids[0], list):
                input_ids = [y for x in input_ids for y in x]
            input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        if self.speech_db:
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat, num_frame = None, 0

        # 获取 img_mask_tgt and img_mask
        img_frames, output_label = self.create_mcrm_io(num_bb, mask_consecutive=self.mask_consecutive)
        img_mask_tgt = _get_img_tgt_mask(output_label, len(input_ids), num_frame)
        img_mask = torch.tensor([True if frame == 1000 else False for frame in img_frames], dtype=torch.long)
        # print(f'[Debug in mspanrc] img mask {img_mask} {len(img_mask)}')
        # 在修改feature之前先获取target.
        # print(f'[Debug in mspanrc] img_soft_labels {img_soft_labels.shape}')
        label_target = _get_target(img_soft_labels, output_label)
        label_target = torch.cat(label_target).reshape(-1, img_soft_labels.size(-1))
        # print(f'[Debug in mspanrc] masked label_target {label_target.shape}')
        assert len(img_mask_tgt) == img_mask_tgt.size(0)
        # 将random replacement 的帧替换一下
        for i, index in enumerate(img_frames):
            if index != 1000 and i != index:
                img_feat[i] = img_feat[index]
                # print(f'[Debug in mspanrc] sample {i} replacement feature indexs {i} to {index}')
        return (input_ids, img_feat, speech_feat,
                img_soft_labels, attn_masks, img_mask, img_mask_tgt, label_target)

    def create_mcrm_io(self, num_bb, mask_consecutive=3):
        input_frames, img_labels = get_consecutive_mask(num_bb, mask_consecutive)
        input_frames = torch.tensor(input_frames)
        img_labels = torch.tensor(img_labels)
        return input_frames, img_labels

def mspanrc_collate(inputs):
    (input_ids, img_feats, speech_feats, img_soft_labels,
     attn_masks, img_masks, img_mask_tgts, label_targets) = map(list, unzip(inputs))

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

    label_targets = torch.cat(label_targets, dim=0).view(-1, img_soft_label.size(-1))
    # print(f'[Debug in mspanrfr] masked batch label_targets {label_targets.shape}')

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


if __name__ == "__main__":
    tokens, output_labels = get_consecutive_mask(20, 3)
    print(f'tokens {tokens}')
    print(f'output {output_labels}')
