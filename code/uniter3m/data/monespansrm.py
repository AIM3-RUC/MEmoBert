"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Mask Span Speech Region Modeling Datasets
直接遮蔽一整个连续的片段: 20% 表示连续遮蔽20%的片段
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
from code.uniter3m.data.monespanrm import get_consecutive_mask

def _get_speech_tgt_mask(output_label, txt_len, img_len):
    '''
    img_frames, img_labels, num_frame
    根据 output_label 来构建img_tgt_mask
    '''
    speech_mask_tgt = [False for i in range(len(output_label))]
    for i in range(len(output_label)):
        if output_label[i] != -1:
            speech_mask_tgt[i] = True
    if float(torch.__version__[:3]) <= 1.2:
        bool_type = torch.uint8
    else:
        bool_type = torch.bool
    speech_mask_tgt = torch.tensor(speech_mask_tgt, dtype=bool_type)
    z = torch.zeros(txt_len, dtype=bool_type)
    if img_len > 0:
        zi = torch.zeros(img_len, dtype=bool_type)
        speech_mask_tgt = torch.cat([z, zi, speech_mask_tgt], dim=0)
    else:
        speech_mask_tgt = torch.cat([z, speech_mask_tgt], dim=0)
    return speech_mask_tgt

def _get_feat_target(one_img_feat, output_label):
    # one_img_feat of one video, 保持与img_tgt_mask一致
    assert len(one_img_feat) == len(output_label)
    feat_target = []
    for i in range(len(output_label)):
        if output_label[i] != -1:
            feat_target.append(one_img_feat[i])
    return feat_target

def _mask_img_feat(img_feat, img_masks):
    # 将masked的部分置为0向量
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked

class MOneSpansrfrDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_len_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        only for speech feature modeling
        '''
        print('MOneSpanrfrDataset span {}'.format(mask_len_ratio))
        self.mask_len_ratio = mask_len_ratio
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

        if self.no_text:
            # 只保留cls分类位置.
            # print('[Debug in MSpansrfrDataset] no text info!!!')
            input_ids = [self.txt_db.cls_]
            input_ids = torch.tensor(input_ids)
        else:
            # text input
            input_ids = example['input_ids']
            if isinstance(input_ids[0], list):
                input_ids = [y for x in input_ids for y in x]
            input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids), dtype=torch.long)
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
        
        # 获取 img_mask_tgt and img_mask
        speech_frames, output_label = self.create_mcrm_io(num_frame, mask_len_ratio=self.mask_len_ratio)
        speech_mask = torch.tensor([True if frame == 1000 else False for frame in speech_frames], dtype=torch.long)
        # print(f'[Debug in mspansrfr] speech mask {speech_mask} {len(speech_mask)}')
        speech_mask_tgt = _get_speech_tgt_mask(output_label, len(input_ids), img_len=num_bb)
        # 在修改feature之前先获取target.
        # print(f'[Debug in mspansrfr] speech feat {speech_feat.shape}')
        feat_target = _get_feat_target(speech_feat, output_label)
        feat_target = torch.cat(feat_target).reshape(-1, speech_feat.size(-1))
        # print(f'[Debug in mspansrfr] masked feat_targets {feat_target.shape}')
        assert sum(speech_mask_tgt) == feat_target.size(0)

        # 将random replacement 的帧替换一下
        for i, index in enumerate(speech_frames):
            if index != 1000 and i != index:
                speech_feat[i] = speech_feat[index]
                # print(f'[Debug in mspansrfr] sample {i} replacement feature indexs {i} to {index}')
        return (input_ids, img_feat, speech_feat, attn_masks, speech_mask, speech_mask_tgt, feat_target)

    def create_mcrm_io(self, num_frame, mask_len_ratio):
        input_frames, img_labels = get_consecutive_mask(num_frame, mask_len_ratio)
        input_frames = torch.tensor(input_frames)
        img_labels = torch.tensor(img_labels)
        return input_frames, img_labels

def monespansrfr_collate(inputs):
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
    (input_ids, img_feats, speech_feats, attn_masks, speech_masks, speech_mask_tgts, feat_targets
     ) = map(list, unzip(inputs))

    # text info
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # speech info
    num_frames = [f.size(0) for f in speech_feats]
    speech_feat = pad_tensors(speech_feats, num_frames)
    speech_position_ids = torch.arange(0, max(num_frames), dtype=torch.long
                                ).unsqueeze(0)

    # mask features
    speech_masks = pad_sequence(speech_masks, batch_first=True, padding_value=0)
    # 不需要pad，只需要将所有masked的target拼接起来即可
    feat_targets = torch.cat(feat_targets, dim=0).view(-1, speech_feat.size(-1))
    # print(f'[Debug in mspansrfr] masked batch feat_targets {feat_targets.shape}')
    speech_feat = _mask_img_feat(speech_feat, speech_masks)
    speech_mask_tgt = pad_sequence(speech_mask_tgts,
                                batch_first=True, padding_value=0)
    assert torch.sum(speech_mask_tgt) == feat_targets.size(0)
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

if __name__ == "__main__":
    tokens, output_labels = get_consecutive_mask(20, 3)
    print(f'tokens {tokens}')
    print(f'output {output_labels}')
