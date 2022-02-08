"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Emo Image Text Matching, EITM
1. 当前文本与自身对应的语音和人脸是情感匹配的 label=1
2. 当前文本与其他的不同情感的人脸和语音是情感不匹配的 label=0
暂时只考虑1和2，相当于挑选了比较任务相关的 negative samples.

3. 当前文本与其他的相同情感的人脸和语音是情感匹配的 label=1 --Pending
"""
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np
import pickle as pkl

from code.uniter3m_mae.data.data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
# from uniter 
from code.uniter.data.itm import TokenBucketSamplerForItm, sample_negative

class EItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling)
     """
    def __init__(self, txt_db, img_db, speech_db, emo2img_fname_path, use_total_eitm, neg_sample_p=0.5):
        '''
        相较与ITM来说，唯一的变化就是改变了负例的筛选策略
        emo2img_fname_path: 需要提供一个根据文本情感来确定
        use_total_eitm: if True 正例也要随意任取
        '''
        assert isinstance(txt_db, TxtTokLmdb)
        if img_db is not None:
            print('[Debug] Img db is not None!!!')
            assert isinstance(img_db, DetectFeatLmdb)
        if speech_db is not None:
            print('[Debug] Speech db is not None!!!')
            assert isinstance(speech_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.speech_db = speech_db
        self.img_shape = None
        self.use_total_eitm = use_total_eitm

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.emo2img_fnames = pkl.load(open(emo2img_fname_path, 'rb'))
        self.emolist = list(self.emo2img_fnames.keys())

        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness
        every epoch to random this
        首先随机生成0or1的labels, 如果label=0, 那么就根据文本构建一个image的负例。
        """
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1-self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            emo_category = super().__getitem__(i)['target']
            emo_category = int(emo_category)
            if self.labels[i] == 0:
                # 随机选择一个除此类别之外的一个情感中的图片作为负面例子
                # print(f'original img frame {img_fname}')
                self.emolist.remove(emo_category)
                # print(f'[Debug in eimt] current emolist {self.emolist}')
                neg_emo = random.sample(self.emolist, 1)[0]
                img_fname = sample_negative(self.emo2img_fnames[neg_emo], [img_fname], 1)[0]
                # print(f'[Debug in eimt] negative sample emo {neg_emo} and img_fname {img_fname}')
                self.emolist = list(self.emo2img_fnames.keys())

            if self.use_total_eitm:
                if self.labels[i] == 1:
                    # 随机选择一个类别的一个情感中的图片作为正面例子
                    # print(f'original postive img frame {img_fname}')
                    img_fname = random.sample(list(self.emo2img_fnames[emo_category]), 1)[0]
                    # print(f'[Debug in eimt] postive sample emo img_fname {img_fname}')

            self.train_imgs.append(img_fname)
            if self.img_db is not None and self.speech_db is None:
                self.lens.append(tl + self.img_db.name2nbb[img_fname])
            if self.speech_db is not None and self.img_db is None :
                self.lens.append(tl + self.speech_db.name2nbb[img_fname])
            if self.speech_db is not None and self.img_db is not None:
                self.lens.append(tl + self.img_db.name2nbb[img_fname] + self.speech_db.name2nbb[img_fname])

    def __getitem__(self, i):
        # i only the index not the real text-id in txtdb
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        # text input
        input_ids = example['input_ids']
        if isinstance(input_ids[0], list):
            input_ids = [y for x in input_ids for y in x]
        # add cls and sep special tokens
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        # must use this, very shit bug that upset me two days
        img_fname = self.train_imgs[i]
        if self.img_db is not None:
            img_feat, num_bb = self._get_img_feat(img_fname, self.img_shape)
            self.img_shape = img_feat.shape[1:]
            img_attn_masks = torch.ones(num_bb, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, img_attn_masks))
        else:
            img_feat = None

        if self.speech_db is not None:
            speech_feat, num_frame = self._get_speech_feat(img_fname)
            speech_attn_masks = torch.ones(num_frame, dtype=torch.long)
            attn_masks = torch.cat((attn_masks, speech_attn_masks))
        else:
            speech_feat = None
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)
        return input_ids, img_feat, speech_feat, attn_masks, target


def eitm_collate(inputs):
    (input_ids, img_feats, speech_feats, attn_masks, targets) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    if img_feats[0] is not None:
        num_bbs = [f.size(0) for f in img_feats]
        img_feat = pad_tensors(img_feats, num_bbs)
        img_position_ids = torch.arange(0, img_feat.size(1), dtype=torch.long
                                    ).unsqueeze(0)
    else:
        img_feat, num_bbs, img_position_ids = None, None, None

    if speech_feats[0] is not None:
        ## speech batches
        num_frames = [f.size(0) for f in speech_feats]
        speech_feat = pad_tensors(speech_feats, num_frames)
        # print('[Debug] the batch input {}'.format(img_feat.shape)) # (n, max_num_nbb, dim)
        speech_position_ids = torch.arange(0, speech_feat.size(1), dtype=torch.long).unsqueeze(0)    
    else:
        speech_feat, num_frames, speech_position_ids = None, None, None

    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, num_frames, bs, max_tl, out_size)

    # print(f'[Debug in itm data] text {input_ids.shape} img {img_feat.shape} speech {speech_feat}')
    # print(f'[Debug in itm data] targets {targets.shape} gather_index {gather_index.shape} attn_masks {attn_masks.shape}')

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_position_ids': img_position_ids,
             'speech_feat': speech_feat,
             'speech_position_ids': speech_position_ids,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch