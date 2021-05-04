"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3flow.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors)
from code.uniter3m.data.data import get_gather_index

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
    def __init__(self, txt_db, img_db, speech_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, speech_db)
        self.img_shape = None

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

        return input_ids, img_feat, speech_feat, text_attn_masks, img_attn_masks, speech_attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        # remove the sep token
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

def mlm_collate(inputs):
    """
    Jinming: modify to img_position_ids
    Return:
    :input_ids    (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :speech_feat     (n, max_num_frames, feat_dim)
    :num_framess     list of [num_frame]
    :img_feat     (n, max_num_bb, feat_dim, feat_dim)
    :num_bbs      list of [num_bb]
    :text_att_masks   (n, max_{L}) padded with 0
    :img_att_masks   (n, max_{num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, speech_feats, text_attn_masks, img_attn_masks, speech_attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)

    # for gather index
    out_size = text_attn_masks.size(1)
    # list of None
    if img_feats[0] is not None:
        img_attn_masks = pad_sequence(img_attn_masks, batch_first=True, padding_value=0)
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs) # (n, max_num_nbb, dim)
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long).unsqueeze(0)
        out_size += img_attn_masks.size(1)
    else:
        num_bbs, img_position_ids, img_feat = None, None, None

    if speech_feats[0] is not None:
        # raw wav and attn degign in model
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames) # (n, max_num_nbb, dim)
        # 关于speech的mask, 采用向下取整的策略, drop last
        speech_attn_masks = pad_sequence(speech_attn_masks, batch_first=True, padding_value=0)
        num_segments = [int(num_frame/(0.02 * 16000) - 1) for num_frame in num_frames]
        out_size += speech_attn_masks.size(1)
        speech_position_ids = torch.arange(0, speech_attn_masks.size(1), dtype=torch.long).unsqueeze(0)
        # 这里需要注意的是，speech-feat 经过 speech encoder的长度可能大于speech_attn_masks，
        # 需要在模型部分做一个截断的后处理.
    else:
        num_segments, speech_position_ids, speech_feat, speech_attn_masks = None, None, None, None

    bs, max_tl = input_ids.size()
    # multi-modality atten mask
    gather_index = get_gather_index(txt_lens, num_bbs, num_segments, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'txt_lens': txt_lens,
            'text_attn_masks': text_attn_masks,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'num_bbs': num_bbs,
             'img_attn_masks': img_attn_masks,
             'speech_feat': speech_feat,
             'num_segments': num_segments,
             'speech_attn_masks': speech_attn_masks,
             'speech_position_ids': speech_position_ids,       
             'gather_index': gather_index,      
             'txt_labels': txt_labels}
    return batch