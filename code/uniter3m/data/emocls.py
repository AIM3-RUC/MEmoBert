"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

training dataset with target
"""
import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, \
                   pad_tensors, get_gather_index, get_gather_index_notxtdb)
                   
class EmoClsDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db=None, speech_db=None, emocls_type='hard', use_text=True, use_emolare=False):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, speech_db)
        # emocls_type: default is hard, use the hard label for downstream tasks
        self.img_shape = None
        self.emocls_type = emocls_type
        self.use_text = use_text
        self.use_emolare = use_emolare

        if not self.use_text and speech_db is None and img_db is None:
            print('[Error] all modalities are None')
            exit(0)

    def __getitem__(self, i):
        """
        Add the condition of that txt_db maybe None
        i: is str type
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)
        if self.emocls_type == 'soft':
            target = example['soft_labels'] # probs
        elif self.emocls_type == 'hard':
            target = example['target']  # int 
        elif self.emocls_type == 'logits':
            target = example['logits']  # logits need to devide by temp in the nodel
        else:
            print('[Error] the error emo classification type')
            exit(0)

        if self.use_text:
            # text input
            input_ids = example['input_ids']
            input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
            attn_masks = torch.ones(len(input_ids), dtype=torch.long)
            if self.use_emolare:
                # text pos 
                input_ids_pos = torch.tensor([4] + example['pos_ids'] + [4])
                input_ids_senti = torch.tensor([2] + example['word_senti_ids'] + [2])
                # text u-senti, use the unknown seq instead at inference stage
                sentence_polarity_ids = [5] * len(input_ids)
            else:
                input_pos_ids, input_senti_ids, sentence_polarity_ids = None, None, None       
        else:
            input_ids, attn_masks, input_pos_ids, input_senti_ids, sentence_polarity_ids = None, None, None, None, None

        img_fname = example['img_fname']
        if self.img_db is not None:
            # print(f'[Debug] item {i} img is not None')
            img_feat, num_bb = self._get_img_feat(img_fname, self.img_shape)
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            self.img_shape = img_feat.shape[1:]
            if attn_masks is None:
                attn_masks = img_attn_masks
            else:
                attn_masks = torch.cat((attn_masks, img_attn_masks))
        else:
            # print(f'[Debug] item img {i} is None')
            img_feat = None
        
        if self.speech_db is not None:
            # print(f'[Debug] item {i} speech is not None')
            speech_feat, num_frame = self._get_speech_feat(img_fname)
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            if attn_masks is None:
                attn_masks = speech_attn_masks
            else:
                attn_masks = torch.cat((attn_masks, speech_attn_masks))
            # print('[Debug] item {} speech attn mask {} and final attn mask {}'.format(i, speech_attn_masks.shape, attn_masks.shape))
        else:
            speech_feat = None

        # for visualization
        frame_name = example['img_fname']
        # print("[Debug empty] txt {} img {}".format(len(input_ids), num_bb))
        return input_ids, img_feat, speech_feat, attn_masks, target, frame_name, input_pos_ids, input_senti_ids, sentence_polarity_ids

def emocls_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len], with cls and sep 
    :img_feat     (n, max_num_bb, feat_dim)
    :img_position_ids (n, max_num_bb)
    :num_bbs      list of [num_bb], real num_bbs
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    """
    (input_ids, img_feats, speech_feats, attn_masks, targets, batch_frame_names, \
                                input_pos_ids, input_senti_ids, sentence_polarity_ids) = map(list, unzip(inputs))
    
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    if input_ids[0] is not None:
        # text batches
        txt_lens = [i.size(0) for i in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)
        if input_pos_ids[0] is not None:
            input_ids_pos = pad_sequence(input_ids_pos, batch_first=True, padding_value=4)
            input_ids_senti = pad_sequence(input_ids_senti, batch_first=True, padding_value=2)
            # input_ids_senti
            sentence_polarity_ids = pad_sequence(sentence_polarity_ids, batch_first=True, padding_value=5)
        else:
            input_ids_pos, input_ids_senti, sentence_polarity_ids = None, None, None
    else:
        txt_lens, input_ids, position_ids = None, None, None
        input_ids_pos, input_ids_senti, sentence_polarity_ids = None, None, None

    if img_feats[0] is not None:
        ## image batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        # print('[Debug] batch padding img input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)      
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

    # 构建gather-indxe
    out_size = attn_masks.size(1)
    # 如果txt是None,那么按照img在前的顺序
    if txt_lens is not None:
        # print('[Debug] use Txt gather')
        bs, max_tl = input_ids.size()
        gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)
    else:
        # print('[Debug] use NoTxt gather!!!')
        if img_feat is None:
            # only have speech modality
            bs, max_bb = speech_feat.size(0), speech_feat.size(1)
        else:
            # have both img and speech two modalities
            bs, max_bb = img_feat.size(0), img_feat.size(1)
        gather_index = get_gather_index_notxtdb(num_bbs, num_frames, bs, max_bb, out_size)

    # transfer targets to tensor (batch-size)
    # print(f'[Debug] EmoCls target {np.array(targets).shape}')
    if len(np.array(targets).shape) == 2:
        # soft-label
        targets = torch.from_numpy(np.array(targets))
    else:
        # hard-label
        targets = torch.from_numpy(np.array(targets).reshape((-1))).long()

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'txt_lens': txt_lens,
             'img_lens': num_bbs,
             'img_frame_names': batch_frame_names,
             'input_ids_pos': input_ids_pos, 
             'input_ids_senti': input_ids_senti, 
             'sentence_polarity_ids': sentence_polarity_ids
             }
    return batch