"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from code.uniter3flow.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors)

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

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
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask
    return tokens, output_label

class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - text_attn_masks   : (num_bb, ), ie., [1, 1, ..., 0, 0]
        - img_attn_masks   : (num_bb, ), ie., [1, 1, ..., 0, 0]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])

        # img input Jinming remove the norm-bbx fts
        img_feat, num_bb = self._get_img_feat(example['img_fname'])

        text_attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        img_attn_masks = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, text_attn_masks, img_attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        # remove the sep token
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids)
        txt_labels = torch.tensor([-1] + txt_labels)
        return input_ids, txt_labels

def mlm_collate(inputs, add_cls_token=False):
    """
    Jinming: modify to img_position_ids
    Return:
    :input_ids    (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :num_bbs      list of [num_bb]
    :text_att_masks   (n, max_{L}) padded with 0
    :img_att_masks   (n, max_{num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :add_cls_token, add cls token or not
    """
    (input_ids, img_feats, text_attn_masks, img_attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)
    if add_cls_token:
        img_position_ids = torch.arange(0, img_feat.size(1)+1, dtype=torch.long).unsqueeze(0)
    else:
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)
    # speech batches

    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)
    img_attn_masks = pad_sequence(img_attn_masks, batch_first=True, padding_value=0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'txt_lens': txt_lens,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'num_bbs': num_bbs,
             'text_attn_masks': text_attn_masks,
             'img_attn_masks': img_attn_masks,
             'txt_labels': txt_labels}
    return batch
