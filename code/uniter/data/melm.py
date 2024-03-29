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
from code.uniter.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors, get_gather_index)

def random_emo_word(melm_prob, tokens, vocab_range, emo_tokens, mask):
    """
    Masking some emotion tokens for 
    :param melm_prob: prob of masked emotional words
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :param emo_tokens: emotional tokens in tokens and the token's emotion
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
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label

class MelmDataset(DetectFeatTxtTokDataset):
    '''
    emotional words masked modeling, the 
    melm_prob: masked probility
    '''
    def __init__(self, mask_prob, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
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
        input_ids, txt_labels = self.create_melm_io(example['input_ids'], example['emo_input_ids'])

        # Jinming: add for emo_type_ids (0~4), 
        # 0 is the no-emotion-words
        if example.get('emo_type_ids') is not None:
            # for emotion type embedding
            emo_type_ids = torch.tensor([0] + example['emo_type_ids'] + [0])
            # generate the labels for multitask emotion, 保持跟 txt_labels 一致，txt_labels 中为 -1 的位置同样置为-1.
            txt_emo_labels = torch.where(txt_labels<0, txt_labels, emo_type_ids)
            # print("[Debug] txt_labels {}".format(txt_labels))
            # print("[Debug] emo_type_ids {}".format(emo_type_ids))
            # print("[Debug] txt_emo_labels {}".format(txt_emo_labels))
        else:
            emo_type_ids = None
            txt_emo_labels = None

        # img input Jinming remove the norm-bbx fts
        img_feat, num_bb = self._get_img_feat(example['img_fname'], self.img_shape)
        self.img_shape = img_feat.shape[1:]

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, attn_masks, txt_labels, emo_type_ids, txt_emo_labels

    def create_melm_io(self, input_ids, emo_input_ids):
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
    (input_ids, img_feats, attn_masks, txt_labels, batch_emo_type_ids, \
             batch_txt_emo_labels) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]

    # Jinming: here emo_type_ids is batch, so judge the element is none or not 
    # batch_emo_type_ids is also can used for 
    if batch_emo_type_ids[0] is not None:
        # print(emo_type_ids)
        batch_emo_type_ids = pad_sequence(batch_emo_type_ids, batch_first=True, padding_value=0)
        batch_txt_emo_labels = pad_sequence(batch_txt_emo_labels, batch_first=True, padding_value=-1)
    else:
        batch_emo_type_ids = None
        batch_txt_emo_labels = None
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    # img_feat = (n, max_num_nbb, dim)
    img_position_ids = torch.arange(0, max(num_bbs), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels, 
             'emo_type_ids': batch_emo_type_ids,
             'txt_emo_labels': batch_txt_emo_labels
             }
    return batch
