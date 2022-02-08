"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MELM datasets: masked emotional language modeling

Update 2020-02-04: Jinming add emo_type_ids
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m_mae.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors, get_gather_index)
from code.uniter.data.mlm import random_word

def random_emo_word(melm_prob, tokens, vocab_range, emo_tokens, mask):
    """
    # 其实有很多的句子都没有mask，首先60%的无情感词的句子，就如果没有任何mask. 所以对于这种情况应该如何处理？
    按照正常的mlm进行处理。
    Masking some emotion tokens for 
    :param melm_prob: prob of masked emotional words
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :param emo_tokens: emotional tokens ids
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        if token in emo_tokens:
            # 80% randomly change token to mask token
            if prob < melm_prob:
                prob /= melm_prob
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = mask
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(range(*vocab_range)))
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # 对于普通句子或者没有 emotion mask的词进行处理, 按照正常的mlm进行处理
        tokens, output_label = random_word(tokens, vocab_range, mask)
        if all(o == -1 for o in output_label):
            output_label[0] = tokens[0]
            tokens[0] = mask
    return tokens, output_label

class MelmDataset(DetectFeatTxtTokDataset):
    '''
    emotional words masked modeling, the 
    melm_prob: masked probility
    '''
    def __init__(self, mask_prob, txt_db, img_db, speech_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, speech_db)
        self.melm_prob = mask_prob
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
        input_ids = example['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]
        input_ids, txt_labels = self.create_melm_io(input_ids, example['emo_input_ids'])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        # Jinming: add for emo_type_ids (0~2), 0 is the no-emotion-words
        # emo_type_ids 是序列长度的情感标签, 每个元素表示当前token的情感类别: 0 1 2 
        if example.get('emo_type_ids') is not None:
            emo_type_ids = torch.tensor([0] + example['emo_type_ids'] + [0])
            # generate the labels for multitask emotion, 保持跟 txt_labels 一致，txt_labels 中为 -1 的位置同样置为-1.
            txt_emo_labels = torch.where(txt_labels<0, txt_labels, emo_type_ids)
            # print("[Debug] txt_labels {}".format(txt_labels))
            # print("[Debug] txt_emo_labels {}".format(txt_emo_labels))
        else:
            txt_emo_labels = None
        
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

        return input_ids, img_feat, speech_feat, attn_masks, txt_labels, txt_emo_labels

    def create_melm_io(self, input_ids, emo_input_ids):
        # emo_input_ids: input_ids 中所有的情感词的 id
        input_ids, txt_labels = random_emo_word(self.melm_prob, input_ids, 
                                                    self.txt_db.v_range, emo_input_ids, self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def melm_collate(inputs):
    """
    Jinming: modify to img_position_ids
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_position_ids (n, max_num_bb)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :emo_type_ids (n, max_L) padded with 0
    :txt_emo_labels (n, max_L) padded with -1, similar with the emo_type_ids
    """
    (input_ids, img_feats, speech_feats, attn_masks, txt_labels, \
             batch_txt_emo_labels) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]

    # Jinming: here emo_type_ids is batch, so judge the element is none or not 
    if batch_txt_emo_labels[0] is not None:
        batch_txt_emo_labels = pad_sequence(batch_txt_emo_labels, batch_first=True, padding_value=-1)
    else:
        batch_txt_emo_labels = None
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

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
             'txt_labels': txt_labels, 
             'txt_emo_labels': batch_txt_emo_labels
             }
    return batch
