"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
ref:
SentiLARE: Sentiment-Aware Language Representation Learning with Linguistic Knowledge
/data7/MEmoBert/code/sentilare/pretrain/data_label.py
区别是, SentiLARE 使用的是 whole word mask，而这里我提前处理成了 input_ids, 为了方便采用的 mask sub-word mask. 应该影响不大.
"""

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from code.uniter3m.data.data import (DetectFeatTxtTokDataset, TxtTokLmdb, pad_tensors, get_gather_index)
from code.uniter.data.mlm import random_word

def random_lare_word(tokens, tokens_pos, tokens_senti, vocab_range, mask):
    """
    默认如果是非情感词, 那么
    Masking some emotion tokens for 
    :param melm_prob: prob of masked emotional words
    :param tokens: list of int, tokenized sentence.
    :param tokens_pos: list of int, tokenized word pos tag， v,a,r,n,other.
    :param tokens_senti: list of int, tokenized word sentiment, 0, 1, 2.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
    """
    output_token_label = []
    output_pos_label = []
    output_senti_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        if tokens_senti[i] == 2:
            # 0-neg, 1-pos, 2-neu 对于非情感词, 正常的跟Bert一样
            if prob < 0.15:
                prob /= 0.15
                ori_pos = tokens_pos[i]
                ori_sen = tokens_senti[i]
                # 80% randomly change token to mask token
                if prob < 0.8:
                    # print('[Debug] 0.8 predict mask token {}'.format(token))
                    tokens[i] = mask
                    tokens_pos[i] = 4 # use index=4 other as mask 
                    tokens_senti[i] = 2 # use index=2 neutral as mask
                # 10% randomly change token to random token
                elif prob < 0.9:
                    # print('[Debug] 0.1 predict random token {}'.format(token))
                    tokens[i] = random.choice(list(range(*vocab_range)))
                    tokens_pos[i] = 4 # use index=4 other as mask 
                    tokens_senti[i] = 2 # use index=2 neutral as mask
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_token_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_token_label.append(-1)
            # 对于普通的词，如果词性是 other的话，不计算这个词的loss.
            if ori_pos != 4:  # the pos_tag of ordinary word is not unknown
                output_pos_label.append(ori_p)
                output_senti_label.append(ori_s)
            else:
                output_pos_label.append(-1)
                output_senti_label.append(-1)
        else:
            if prob < 0.3:
                prob /= 0.3
                ori_pos = tokens_pos[i]
                ori_sen = tokens_senti[i]
                # 80% randomly change token to mask token
                if prob < 0.8:
                    # print('[Debug] 0.8 predict mask token {}'.format(token))
                    tokens[i] = mask
                    tokens_pos[i] = 4 # use index=4 other as mask 
                    tokens_senti[i] = 2 # use index=2 neutral as mask
                # 10% randomly change token to random token
                elif prob < 0.9:
                    # print('[Debug] 0.1 predict random token {}'.format(token))
                    tokens[i] = random.choice(list(range(*vocab_range)))
                    tokens_pos[i] = 4 # use index=4 other as mask 
                    tokens_senti[i] = 2 # use index=2 neutral as mask
                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                # 对于所有mask的情感词都采用 pos 和 senti 的预测
                output_token_label.append(token)
                output_pos_label.append(ori_p)
                output_senti_label.append(ori_s)
            else:
                # no masking token (will be ignored by loss function later)
                output_token_label.append(-1)
                output_pos_label.append(-1)
                output_senti_label.append(-1)  
    if all(o == -1 for o in output_label):
        output_token_label[0] = tokens[0]
        output_pos_label[0] = tokens[0]
        output_senti_label[0] = tokens[0]
        tokens[0] = mask
        tokens_pos[0] = 4
        tokens_senti[0] = 2

    return tokens, tokens_pos, tokens_senti, output_token_label, output_pos_label, output_senti_label

class EmoLareDataset(DetectFeatTxtTokDataset):
    '''
    emotional words masked modeling, the 
    melm_prob: masked probility
    '''
    def __init__(self, task_ratio, txt_db, img_db, speech_db):
        # task_ratio: a ration for late Supervision and others for early fusion
        # can to try [0.2 0.4 0.6]
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, speech_db)
        self.task_ratio = task_ratio
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
        input_ids = example['input_ids']
        input_pos_ids = example['pos_ids']
        input_senti_ids = example['word_senti_ids']
        target = example['target'] # utterance-level label

        # text input
        input_ids, input_ids_pos, input_ids_senti, txt_labels, txt_pos_labels, txt_sen_labels = self.create_lare_io(
                        input_ids, input_pos_ids, input_senti_ids)
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        # for sentence-level task
        prob = random.random()
        if prob < self.task_ratio:
            # Late Supervision, 输入全部设置为 unknown type
            sentence_polarity_ids = [5] * len(input_ids)
            sentence_polarity_label = [target] + [-1] * (len(input_ids_pos) - 1)
        else:
            # Early Fusion
            sentence_polarity_ids = [5] + [target] * (len(input_ids)-2) + [5]
            sentence_polarity_label = [-1] * len(input_ids_pos)
        
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

        return input_ids, img_feat, speech_feat, attn_masks, txt_labels, input_ids_pos, txt_pos_labels, input_ids_senti, txt_sen_labels, sentence_polarity_ids, sentence_polarity_label

    def create_lare_io(self, input_ids, tokens_pos, tokens_senti):
        # emo_input_ids: input_ids 中所有的情感词的 id
        input_ids, tokens_pos, tokens_senti, output_token_labels, output_pos_labels, output_senti_labels = random_lare_word(input_ids, 
                                                tokens_pos, tokens_senti, self.txt_db.v_range, self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])
        input_ids_pos = torch.tensor([4] + tokens_pos + [4])
        input_ids_senti = torch.tensor([2] + tokens_senti + [2])
        txt_labels = torch.tensor([-1] + output_token_labels + [-1])
        txt_pos_labels = torch.tensor([-1] + output_pos_labels + [-1])
        txt_sen_labels = torch.tensor([-1] + output_senti_labels + [-1])
        return input_ids, input_ids_pos, input_ids_senti, txt_labels, txt_pos_labels, txt_sen_labels


def emolare_collate(inputs):
    """
    OK
    """
    (input_ids, img_feats, speech_feats, attn_masks, txt_labels, input_ids_pos, txt_pos_labels, \
        input_ids_senti, txt_senti_labels, sentence_polarity_ids, sentence_polarity_label) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]

    # Jinming: here emo_type_ids is batch, so judge the element is none or not 
    if batch_txt_emo_labels[0] is not None:
        batch_txt_emo_labels = pad_sequence(batch_txt_emo_labels, batch_first=True, padding_value=-1)
    else:
        batch_txt_emo_labels = None
    # input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    # input_ids_pos
    input_ids_pos = pad_sequence(input_ids_pos, batch_first=True, padding_value=4)
    txt_pos_labels = pad_sequence(txt_pos_labels, batch_first=True, padding_value=-1)
    # input_ids_senti
    input_ids_senti = pad_sequence(input_ids_senti, batch_first=True, padding_value=2)
    txt_senti_labels = pad_sequence(txt_senti_labels, batch_first=True, padding_value=-1)
    # input_ids_senti
    sentence_polarity_ids = pad_sequence(sentence_polarity_ids, batch_first=True, padding_value=5)
    sentence_polarity_label = pad_sequence(sentence_polarity_label, batch_first=True, padding_value=-1)

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
             'input_ids_pos': input_ids_pos, 
             'txt_pos_labels': txt_pos_labels, 
             'input_ids_senti': input_ids_senti, 
             'txt_senti_labels': txt_senti_labels, 
             'sentence_polarity_ids': sentence_polarity_ids, 
             'sentence_polarity_label': sentence_polarity_label
             }
    return batch