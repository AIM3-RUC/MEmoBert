"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

given the input_ids(with mask) and text_labels, just to predict the masked labels(emotion category)
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m_mae.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)

class PromptMaskDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, speech_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, speech_db)
        self.img_shape = None

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)
        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'], example['text_labels'])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        # print('[Debug] item {} text attn mask {} {}'.format(i, attn_masks, attn_masks.shape))

        if self.img_db is not None:
            # print(f'[Debug] item {i} img is not None')
            img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            self.img_shape = img_feat.shape[1:]
            attn_masks = torch.cat((attn_masks, img_attn_masks))
        else:
            # print(f'[Debug] item img {i} is None')
            img_feat = None
        
        if self.speech_db is not None:
            # print(f'[Debug] item {i} speech is not None')
            speech_feat, num_frame = self._get_speech_feat(example['img_fname'])
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
            # print('[Debug] item {} speech attn mask {} and final attn mask {}'.format(i, speech_attn_masks.shape, attn_masks.shape))
        else:
            speech_feat = None

        return input_ids, img_feat, speech_feat, attn_masks, txt_labels, i
        
    def create_mlm_io(self, input_ids, text_labels):
        # 展开成token的形式
        flat_input_ids, flat_text_labels = [], []
        for sub_input_ids, sub_text_labels in zip(input_ids, text_labels):
            flat_input_ids.extend(sub_input_ids)
            flat_text_labels.extend(sub_text_labels)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + flat_input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + flat_text_labels + [-1])
        return input_ids, txt_labels

def prompt_mask_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_position_ids (n, max_num_bb)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, speech_feats, attn_masks, txt_labels, example_idxs) = map(list, unzip(inputs))
    # print(f'[Debug] batch indexs {example_idxs}')
    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    # multimodality atten mask
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # print('[Debug] batch attn_masks {} {}'.format(attn_masks, attn_masks.shape))
    # print('[Debug] batch padding input_ids {} {}'.format(input_ids, input_ids.shape))

    if img_feats[0] is not None:
        ## image batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        # print('[Debug] batch padding img input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
        img_position_ids = torch.arange(0, max(num_bbs), dtype=torch.long).unsqueeze(0)      
    else:
        img_feat, num_bbs, img_position_ids = None, None, None
    
    if speech_feats[0] is not None:
        ## speech batches
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames)
        # print('[Debug] batch padding speech input {}'.format(speech_feat.shape)) # (n, max_num_frame, dim)
        speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0)    
    else:
        speech_feat, num_frames, speech_position_ids = None, None, None

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    # multi-modality atten mask
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)
    # print('[Debug] batch gather_index {} {}'.format(gather_index, gather_index.shape))
    
    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch