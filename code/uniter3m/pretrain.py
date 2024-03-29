"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER pre-training
"""
import argparse
from collections import defaultdict
from io import BufferedIOBase
import json
from logging import Logger
import os
from os.path import exists, join
from time import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from code.uniter3m.data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, SpeechLmdbGroup, ConcatDatasetWithLens,
                  MlmDataset, MelmDataset, MrfrDataset, MrcDataset,
                  mlm_collate, melm_collate, mrfr_collate, mrc_collate,
                  ItmDataset, itm_collate, MsrfrDataset, msrfr_collate,
                  EmoClsDataset, emocls_collate,
                  EmoLareDataset, emolare_collate,
                  EItmDataset, eitm_collate,
                  OneModalNegItmDataset, onemodal_negitm_collate,
                  MerfrDataset, MercDataset,
                  MSpanrfrDataset, mspanrfr_collate, MSpanrcDataset, mspanrc_collate,
                  MSpansrfrDataset, mspansrfr_collate,
                  MOneSpanrfrDataset, monespanrfr_collate, MOneSpanrcDataset, monespanrc_collate,
                  MOneSpansrfrDataset, monespansrfr_collate,
                  MlmWWMDataset, mlm_wwm_collate,
                  VFOMDataset, vfom_collate,
                  SFOMDataset, sfom_collate,
                  MelmWWMDataset, melm_wwm_collate,
                  PromptMaskDataset, prompt_mask_collate,
                  CrossModalPromptMaskDataset,
                  PromptNSPDataset, prompt_nsp_collate,
                  FlexPromptMaskDataset, flexprompt_mask_collate, CrossModalFlexPromptMaskDataset,
                  FlexPromptMissMaskDataset, flexpromptmiss_mask_collate,
                  )
from code.uniter3m.model.pretrain import UniterForPretraining
from code.uniter3m.optim.misc import build_optimizer
from code.uniter3m.model.emocls import evaluation_metric

# from uniter
from code.uniter.optim import get_lr_sched
from code.uniter.utils.const import IMG_LABEL_DIM, BUCKET_SIZE
from code.uniter.utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from code.uniter.utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from code.uniter.utils.save import ModelSaver, save_training_meta
from code.uniter.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from code.uniter.pretrain import build_dataloader, build_dataloader_itm

def build_mlm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MlmDataset(t, i, s, mask_prob=opts.mlm_prob) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MlmDataset(t, None, s, mask_prob=opts.mlm_prob) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MlmDataset(t, i, None, mask_prob=opts.mlm_prob) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in mlm dataset] the img and speech modality are None!')
            datasets = [MlmDataset(t, None, None, mask_prob=opts.mlm_prob) for t in txt_db]
        else:
            LOGGER.info('[Error] Error mlm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MlmDataset(txt_db, img_db, speech_db, mask_prob=opts.mlm_prob)
    return dataset, mlm_collate

def build_mlm_wwm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MlmWWMDataset(t, i, s, mask_prob=opts.mlm_prob) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MlmWWMDataset(t, None, s, mask_prob=opts.mlm_prob) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MlmWWMDataset(t, i, None, mask_prob=opts.mlm_prob) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in mlm dataset] the img and speech modality are None!')
            datasets = [MlmWWMDataset(t, None, None, mask_prob=opts.mlm_prob) for t in txt_db]
        else:
            LOGGER.info('[Error] Error mlm_wwm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MlmWWMDataset(txt_db, img_db, speech_db, mask_prob=opts.mlm_prob)
    return dataset, mlm_wwm_collate


def build_melm_wwm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MelmWWMDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MelmWWMDataset(t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MelmWWMDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in mlm dataset] the img and speech modality are None!')
            datasets = [MelmWWMDataset(t, None, None) for t in txt_db]
        else:
            LOGGER.info('[Error] Error mlm_wwm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MelmWWMDataset(txt_db, img_db, speech_db)
    return dataset, melm_wwm_collate

def build_melm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MelmDataset(opts.melm_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MelmDataset(opts.melm_prob, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MelmDataset(opts.melm_prob, t, i, None) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            # LOGGER.info('[Debug in melm dataset] the img and speech modality are None!')
            datasets = [MelmDataset(opts.melm_prob, t, None, None) for t in txt_db]
        else:
            LOGGER.info('[Error] Error melm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MelmDataset(opts.melm_prob, txt_db, img_db, speech_db)
    return dataset, melm_collate

def build_emolare_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [EmoLareDataset(opts.emolare_LStask_ratio, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [EmoLareDataset(opts.emolare_LStask_ratio, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [EmoLareDataset(opts.emolare_LStask_ratio, t, i, None) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error emolare datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = EmoLareDataset(opts.emolare_LStask_ratio, txt_db, img_db, speech_db)

    return dataset, emolare_collate

def build_mrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MrfrDataset(opts.mrm_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MrfrDataset(opts.mrm_prob, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in mrfr dataset] the text and speech modality are None!')
            datasets = [MrfrDataset(opts.mrm_prob, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error mrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MrfrDataset(opts.mrm_prob, txt_db, img_db, speech_db)

    return dataset, mrfr_collate

def build_merfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    # use the mrfr collection function
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MerfrDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MerfrDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in merfr dataset] the text and speech modality are None!')
            datasets = [MerfrDataset(None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error merfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MerfrDataset(txt_db, img_db, speech_db)

    return dataset, mrfr_collate

def build_mspanrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    # use the span mrfr collection function
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MSpanrfrDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MSpanrfrDataset(opts.mask_visual_len_ratio,  opts.mask_visual_consecutive, opts.mixed_ratios, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in mspanrfr dataset] the text and speech modality are None!')
            datasets = [MSpanrfrDataset(opts.mask_visual_len_ratio,  opts.mask_visual_consecutive, opts.mixed_ratios,  None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error mspanrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpanrfrDataset(opts.mask_visual_len_ratio,  opts.mask_visual_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db)
    return dataset, mspanrfr_collate

def build_monespanrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    # use the one span mrfr collection function
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MOneSpanrfrDataset(opts.mask_visual_len_ratio, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MOneSpanrfrDataset(opts.mask_visual_len_ratio, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in monespanrfr dataset] the text and speech modality are None!')
            datasets = [MOneSpanrfrDataset(opts.mask_visual_len_ratio, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error monespanrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MOneSpanrfrDataset(opts.mask_visual_len_ratio, txt_db, img_db, speech_db)
    return dataset, monespanrfr_collate

def build_vfom_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if txt_db is not None and speech_db is not None:
            datasets = [VFOMDataset(opts.vfom_random_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif txt_db is not None and speech_db is None:
            datasets = [VFOMDataset(opts.vfom_random_prob, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in vfom dataset] the text and speech modality are None!')
            datasets = [VFOMDataset(opts.vfom_random_prob, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error vfom datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = VFOMDataset(opts.vfom_random_prob, txt_db, img_db, speech_db)
    return dataset, vfom_collate

def build_sfom_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert speech_db != None
    if is_train:
        if txt_db is not None and img_db is not None:
            datasets = [SFOMDataset(opts.sfom_random_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif txt_db is not None and img_db is None:
            datasets = [SFOMDataset(opts.sfom_random_prob, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif txt_db is None and img_db is None:
            LOGGER.info('[Debug in sfom dataset] the text and img modality are None!')
            datasets = [SFOMDataset(opts.sfom_random_prob, None, None, s) for s in speech_db]
        else:
            LOGGER.info('[Error] Error sfom datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = SFOMDataset(opts.sfom_random_prob, txt_db, img_db, speech_db)
    return dataset, sfom_collate

def build_msrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert speech_db != None
    if is_train:
        if img_db is not None:
            datasets = [MsrfrDataset(opts.msrm_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None:
            datasets = [MsrfrDataset(opts.msrm_prob, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif txt_db is None and img_db is None:
            LOGGER.info('[Debug in msrfr dataset] the text and img modality are None!')
            datasets = [MrfrDataset(opts.msrm_prob, None, None, s) for s in speech_db]
        else:
            LOGGER.info('[Error] Error msrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MsrfrDataset(opts.msrm_prob, txt_db, img_db, speech_db)

    return dataset, msrfr_collate

def build_mspansrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert speech_db != None
    if is_train:
        if img_db is not None:
            datasets = [MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None:
            datasets = [MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif txt_db is None and img_db is None:
            LOGGER.info('[Debug in mspansrfr dataset] the text and img modality are None!')
            datasets = [MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios,  None, None, s) for s in speech_db]
        else:
            LOGGER.info('[Error] Error mspansrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db)
    return dataset, mspansrfr_collate

def build_monespansrfr_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert speech_db != None
    if is_train:
        if img_db is not None:
            datasets = [MOneSpansrfrDataset(opts.mask_speech_len_ratio, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None:
            datasets = [MOneSpansrfrDataset(opts.mask_speech_len_ratio, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif txt_db is None and img_db is None:
            LOGGER.info('[Debug in mspansrfr dataset] the text and img modality are None!')
            datasets = [MOneSpansrfrDataset(opts.mask_speech_len_ratio, None, None, s) for s in speech_db]
        else:
            LOGGER.info('[Error] Error mspansrfr datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MOneSpansrfrDataset(opts.mask_speech_len_ratio, txt_db, img_db, speech_db)
    return dataset, monespansrfr_collate

def build_mspansrfrnotext_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert speech_db != None
    if is_train:
        if img_db is not None:
            datasets = [MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, t, i, s, no_text=True) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None:
            datasets = [MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, t, None, s, no_text=True) for t, s in zip(txt_db, speech_db)]
        else:
            LOGGER.info('[Error] Error mspansrfrnotext datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpansrfrDataset(opts.mask_speech_len_ratio, opts.mask_speech_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db, no_text=True)

    return dataset, mspansrfr_collate

def build_mrc_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MrcDataset(opts.mrm_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MrcDataset(opts.mrm_prob, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in mrc dataset] the text and speech modality are None!')
            datasets = [MrfrDataset(opts.mrm_prob, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error mrc datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MrcDataset(opts.mrm_prob, txt_db, img_db, speech_db)

    return dataset, mrc_collate

def build_merc_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MercDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MercDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in merckl dataset] the text and speech modality are None!')
            datasets = [MrfrDataset(None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error merc datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MercDataset(txt_db, img_db, speech_db)
    return dataset, mrc_collate

def build_mspanrc_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in mspanrc dataset] the text and speech modality are None!')
            datasets = [MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error mspanrc datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db)
    return dataset, mspanrc_collate

def build_monespanrc_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MOneSpanrcDataset(opts.mask_visual_len_ratio, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MOneSpanrcDataset(opts.mask_visual_len_ratio, t, i, None) for t, i in zip(txt_db, img_db)]
        elif txt_db is None and speech_db is None:
            LOGGER.info('[Debug in mspanrc dataset] the text and speech modality are None!')
            datasets = [MOneSpanrcDataset(opts.mask_visual_len_ratio, None, i, None) for i in img_db]
        else:
            LOGGER.info('[Error] Error mspanrc datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MOneSpanrcDataset(opts.mask_visual_len_ratio, txt_db, img_db, speech_db)
    return dataset, monespanrc_collate

def build_mspanrfrnotext_dataset(txt_db, img_db, speech_db, is_train, opts):
    # use the mrfr collection function
    # 提供 textdb 只是为了获取 cls 的信息
    assert img_db != None
    if is_train:
        if speech_db is not None:
            LOGGER.info('[Info] only both speech and img modaity')
            datasets = [MSpanrfrDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, s, no_text=True) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MSpanrfrDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, None, no_text=True) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error mspanrfrnotext datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpanrfrDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db, no_text=True)
    return dataset, mspanrfr_collate

def build_mspanrcnotext_dataset(txt_db, img_db, speech_db, is_train, opts):
    assert img_db != None
    if is_train:
        if speech_db is not None:
            datasets = [MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, t, i, s, no_text=True) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif speech_db is None:
            datasets = [MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive,opts.mixed_ratios,  t, i, None, no_text=True) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error mspanrcklnotext datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MSpanrcDataset(opts.mask_visual_len_ratio, opts.mask_visual_consecutive, opts.mixed_ratios, txt_db, img_db, speech_db, no_text=True)
    return dataset, mspanrc_collate

def build_emocls_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [EmoClsDataset(t, i, s, opts.emocls_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [EmoClsDataset(t, None, s, opts.emocls_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [EmoClsDataset(t, i, None, opts.emocls_type) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in emocls dataset] the img and speech modality are None!')
            datasets = [EmoClsDataset(t, None, None, opts.emocls_type) for t in txt_db]
        else:
            LOGGER.info('[Error] Error itm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = EmoClsDataset(txt_db, img_db, speech_db, opts.emocls_type)
    collate_fn = emocls_collate
    return dataset, collate_fn

def build_itm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [ItmDataset(t, i, s, opts.itm_neg_prob) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [ItmDataset(t, None, s, opts.itm_neg_prob) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [ItmDataset(t, i, None, opts.itm_neg_prob) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error itm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, img_db, speech_db, opts.itm_neg_prob)
    collate_fn = itm_collate
    return dataset, collate_fn

def build_onemodalnegitm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [OneModalNegItmDataset(t, i, s, opts.itm_neg_prob, opts.itm_neg_img_prob) for t, i, s in zip(txt_db, img_db, speech_db)]
        else:
            LOGGER.info('[Error] Error itm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = OneModalNegItmDataset(txt_db, img_db, speech_db, opts.itm_neg_prob, opts.itm_neg_img_prob)
    collate_fn = onemodal_negitm_collate
    return dataset, collate_fn

def build_vtm_dataset(txt_db, img_db, is_train, opts):
    # only consider the visual and txt matching
    if is_train:
        datasets = [ItmDataset(t, i, None, opts.itm_neg_prob) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, img_db, None, opts.itm_neg_prob)
    collate_fn = itm_collate
    return dataset, collate_fn

def build_stm_dataset(txt_db, speech_db, is_train, opts):
    # only consider the speech and txt matching
    if is_train:
        datasets = [ItmDataset(t, None, s, opts.itm_neg_prob) for t, s in zip(txt_db, speech_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, None, speech_db, opts.itm_neg_prob)
    collate_fn = itm_collate
    return dataset, collate_fn

def build_eitm_dataset(txt_db, img_db, speech_db, emo2img_fname_path, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [EItmDataset(t, i, s, p, opts.use_total_eitm, opts.itm_neg_prob) for t, i, s, p in zip(txt_db, img_db, speech_db, emo2img_fname_path)]
        elif img_db is None and speech_db is not None:
            datasets = [EItmDataset(t, None, s, p, opts.use_total_eitm, opts.itm_neg_prob) for t, s, p in zip(txt_db, speech_db, emo2img_fname_path)]
        elif img_db is not None and speech_db is None:
            datasets = [EItmDataset(t, i, None, p, opts.use_total_eitm, opts.itm_neg_prob) for t, i, p in zip(txt_db, img_db, emo2img_fname_path)]
        else:
            LOGGER.info('[Error] Error eitm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = EItmDataset(txt_db, img_db, speech_db, emo2img_fname_path[0], opts.use_total_eitm, opts.itm_neg_prob)
    collate_fn = eitm_collate
    return dataset, collate_fn

def build_prompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [PromptMaskDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [PromptMaskDataset(t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [PromptMaskDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in promptmask dataset] the img and speech modality are None!')
            datasets = [PromptMaskDataset(t, None, None) for t in txt_db]
        else:
            LOGGER.info('[Error] Error promptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = PromptMaskDataset(txt_db, img_db, speech_db)
    return dataset, prompt_mask_collate

def build_cm_prompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [CrossModalPromptMaskDataset(t, i, s, prompt_type=opts.prompt_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [CrossModalPromptMaskDataset(t, None, s, prompt_type=opts.prompt_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [CrossModalPromptMaskDataset(t, i, None, prompt_type=opts.prompt_type) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error promptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = CrossModalPromptMaskDataset(txt_db, img_db, speech_db, prompt_type=opts.prompt_type)
    return dataset, prompt_mask_collate

def build_flexprompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [FlexPromptMaskDataset(t, i, s, prompt_type=opts.prompt_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [FlexPromptMaskDataset(t, None, s, prompt_type=opts.prompt_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [FlexPromptMaskDataset(t, i, None, prompt_type=opts.prompt_type) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in flexpromptmask dataset] the img and speech modality are None!')
            datasets = [FlexPromptMaskDataset(t, None, None, prompt_type=opts.prompt_type) for t in txt_db]
        else:
            LOGGER.info('[Error] Error flexpromptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = FlexPromptMaskDataset(txt_db, img_db, speech_db, prompt_type=opts.prompt_type)
    return dataset, flexprompt_mask_collate

def build_cm_flexprompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [CrossModalFlexPromptMaskDataset(t, i, s, prompt_type=opts.prompt_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [CrossModalFlexPromptMaskDataset(t, None, s, prompt_type=opts.prompt_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [CrossModalFlexPromptMaskDataset(t, i, None, prompt_type=opts.prompt_type) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error cmflexpromptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = CrossModalFlexPromptMaskDataset(txt_db, img_db, speech_db, prompt_type=opts.prompt_type)
    return dataset, flexprompt_mask_collate

def build_flexpromptmiss_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [FlexPromptMissMaskDataset(t, i, s, use_text=True, prompt_type=opts.prompt_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [FlexPromptMissMaskDataset(t, None, s, use_text=True,  prompt_type=opts.prompt_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [FlexPromptMissMaskDataset(t, i, None, use_text=True,  prompt_type=opts.prompt_type) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in flexpromptmask dataset] the img and speech modality are None!')
            datasets = [FlexPromptMissMaskDataset(t, None, None, use_text=True, prompt_type=opts.prompt_type) for t in txt_db]
        else:
            LOGGER.info('[Error] Error flexpromptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = FlexPromptMissMaskDataset(txt_db, img_db, speech_db, use_text=True,  prompt_type=opts.prompt_type)
    return dataset, flexpromptmiss_mask_collate

def build_cm_flexpromptmiss_mask_dataset(txt_db, img_db, speech_db, is_train, opts):
    # use_text: use text modality or not.
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [FlexPromptMissMaskDataset(t, i, s, use_text=False, prompt_type=opts.prompt_type) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [FlexPromptMissMaskDataset(t, None, s, use_text=False, prompt_type=opts.prompt_type) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [FlexPromptMissMaskDataset(t, i, None, use_text=False, prompt_type=opts.prompt_type) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error cmflexpromptmask mask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = FlexPromptMissMaskDataset(txt_db, img_db, speech_db, use_text=False, prompt_type=opts.prompt_type)
    return dataset, flexpromptmiss_mask_collate

def build_prompt_nsp_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [PromptNSPDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [PromptNSPDataset(t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [PromptNSPDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in promptmask nsp dataset] the img and speech modality are None!')
            datasets = [PromptNSPDataset(t, None, None) for t in txt_db]
        else:
            LOGGER.info('[Error] Error promptmask datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = PromptNSPDataset(txt_db, img_db, speech_db)
    return dataset, prompt_nsp_collate


def create_dataloaders(datasets, is_train, opts, all_img_dbs=None, all_speech_dbs=None):
    if all_img_dbs is None and opts.use_visual:
        LOGGER.info('[Debug] Use ImageLmdbGroup')
        all_img_dbs = ImageLmdbGroup(opts.compressed_db)
    
    if all_speech_dbs is None and opts.use_speech:
        LOGGER.info('[Debug] Use SpeechLmdbGroup')
        all_speech_dbs = SpeechLmdbGroup(opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            if  dset.get('img') is not None and opts.use_visual:
                assert len(dset['db']) == len(dset['img'])
                img_db = [all_img_dbs[path] for path in dset['img']]
            else:
                img_db = None
            if dset.get('speech') is not None and opts.use_speech:
                assert len(dset['db']) == len(dset['speech'])
                speech_db = [all_speech_dbs[path] for path in dset['speech']]
            else:
                speech_db = None
        else:
            if dset.get('img') is not None and opts.use_visual:
                assert len(dset['db']) == len(dset['img'])
                img_db = all_img_dbs[dset['img'][0]]
            else:
                img_db = None
            if dset.get('speech') is not None and opts.use_speech:
                speech_db = all_speech_dbs[dset['speech'][0]]
            else:
                speech_db = None

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'
            if is_train:
                LOGGER.info(f"Loading {task} train dataset {dset['db']}")
                txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                          for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} train dataset {dset['db']}")
                txt_db = TxtTokLmdb(dset['db'][0], max_txt_len=60)

            if task.startswith('mlm_wwm'):
                # 由于 mlm_wwm 和 mlm 都是mlm开头，所以这里的顺序不能换
                dataset = build_mlm_wwm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mlm'):
                dataset = build_mlm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('melm_wwm'):
                # 由于 melm_wwm 和 melm 都是melm开头，所以这里的顺序不能换
                dataset = build_melm_wwm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('melm'):
                dataset = build_melm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mrfr'):
                dataset = build_mrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('msrfr'):
                dataset = build_msrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mrc'):
                dataset = build_mrc_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('merfr'):
                dataset = build_merfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('merc'):
                dataset = build_merc_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('vfom'):
                dataset = build_vfom_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('sfom'):
                dataset = build_sfom_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspanrfrnotext'):
                dataset = build_mspanrfrnotext_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspanrcklnotext'):
                dataset = build_mspanrcnotext_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspansrfrnotext'):
                dataset = build_mspansrfrnotext_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspanrfr'):
                dataset = build_mspanrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspanrc'):
                dataset = build_mspanrc_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('mspansrfr'):
                dataset = build_mspansrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('monespanrfr'):
                dataset = build_monespanrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('monespanrc'):
                dataset = build_monespanrc_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('monespansrfr'):
                dataset = build_monespansrfr_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('emocls'):
                dataset = build_emocls_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('itm'):
                dataset = build_itm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('stm'):
                dataset = build_stm_dataset(txt_db, speech_db, is_train, opts)
            elif task.startswith('vtm'):
                dataset = build_vtm_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('eitm'):
                emo2img_fname_path = dset['emo2imgfname']
                LOGGER.info(f'[Info] emo2img_fname_path {emo2img_fname_path}')
                dataset = build_eitm_dataset(txt_db, img_db, speech_db, emo2img_fname_path, is_train, opts)
            elif task.startswith('emolare'):
                dataset = build_emolare_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('onemodalnegitm'):
                dataset = build_onemodalnegitm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('promptmask'):
                dataset = build_prompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('cmpromptmask'):
                dataset = build_cm_prompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('flexpromptmissmask'):
                dataset = build_flexpromptmiss_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('cmflexpromptmissmask'):
                dataset = build_cm_flexpromptmiss_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('flexpromptmask'):
                dataset = build_flexprompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('cmflexpromptmask'):
                dataset = build_cm_flexprompt_mask_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('promptnsp'):
                dataset = build_prompt_nsp_dataset(txt_db, img_db, speech_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task} of dataloader')

            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            if task.startswith('itm') or task.startswith('onemodalnegitm'):
                # itm handles distributed training in dset not sampler
                loader = build_dataloader_itm(*dataset, is_train, opts)
            elif task.startswith('vtm'):
                loader = build_dataloader_itm(*dataset, is_train, opts)
            elif task.startswith('stm'):
                loader = build_dataloader_itm(*dataset, is_train, opts)
            elif task.startswith('eitm'):
                loader = build_dataloader_itm(*dataset, is_train, opts)
            else:
                loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders, all_img_dbs, all_speech_dbs


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps, initial=opts.checkpoint_step)
        model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    all_dbs = [db for datasets in [opts.train_datasets, opts.val_datasets]
               for dset in datasets for db in dset['db']]
    meta_data = json.load(open(f'{all_dbs[0]}/meta.json'))
    if meta_data.get('bert') is None:
        tokenizer = meta_data['tokenizer']
        # print(tokenizer)
        # print([json.load(open(f'{db}/meta.json'))['tokenizer']for db in all_dbs])
    else:
        tokenizer = meta_data['bert']
    # build data loaders
    train_dataloaders, _ , _ = create_dataloaders(
        opts.train_datasets, True, opts)
    val_dataloaders, _, _ = create_dataloaders(
        opts.val_datasets, False, opts)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    if opts.checkpoint:
        LOGGER.info('[Info] Loading from pretrained model {}'.format(opts.checkpoint))
        checkpoint = torch.load(opts.checkpoint)
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if 'classifier' not in k:
                new_checkpoint[k] = v
        checkpoint = new_checkpoint
    else:
        LOGGER.info('[Info] Loading None pretrained model')
        checkpoint = {}
    model = UniterForPretraining.from_pretrained(
        opts.model_config, checkpoint, img_dim=opts.IMG_DIM, speech_dim=opts.Speech_DIM, 
        img_label_dim=IMG_LABEL_DIM, use_visual=opts.use_visual, use_speech=opts.use_speech,
        frozen_en_layers=opts.frozen_en_layers)
        
    model.to(device)
    model.train()
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    model, optimizer = amp.initialize(model, optimizer,
                                      num_losses=len(task2scaler),
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    if opts.checkpoint_step > 0:
        global_step = opts.checkpoint_step
        LOGGER.info("Continue train begin at {}".format(global_step))

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}
    LOGGER.info(f'[Debug] {task2loss}')
    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    for step, (name, batch) in enumerate(meta_loader):
        # LOGGER.info(f'[Debug] train on step: {step}')
        n_examples[name] += batch['input_ids'].size(0)
        n_in_units[name] += (batch['attn_masks'] == 1).sum().item()
        task = name.split('_')[0]
        loss = model(batch, task=task, compute_loss=True)

        n_loss_units[name] += loss.size(0)
        loss = loss.mean()  # loss is not normalized in model
        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))
        task2loss[name](loss.item())

        # optimizer update and logging
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1
            # learning rate scheduling
            if opts.is_reinit_lr:
                lr_this_step = get_lr_sched(global_step-opts.checkpoint_step, opts.lr_sched_type, opts)
            else:
                lr_this_step = get_lr_sched(global_step, opts.lr_sched_type, opts)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            # NOTE: not gathered across GPUs for efficiency
            TB_LOGGER.log_scaler_dict({l.name: l.val
                                       for l in task2loss.values()
                                       if l.val is not None})
            TB_LOGGER.step()

            # update model params
            if opts.grad_norm != -1:
                grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                            opts.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            if global_step % 200 == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                LOGGER.info('Current learning rate {}'.format(lr_this_step))
                for t in train_dataloaders.keys():
                    assert all(tt == t for tt in all_gather_list(t))
                    tot_ex = sum(all_gather_list(n_examples[t]))
                    ex_per_sec = int(tot_ex / (time()-start))
                    tot_in = sum(all_gather_list(n_in_units[t]))
                    in_per_sec = int(tot_in / (time()-start))
                    tot_l = sum(all_gather_list(n_loss_units[t]))
                    l_per_sec = int(tot_l / (time()-start))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info(f'===============================================')

            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'Step {global_step}: start validation')
                validate(model, val_dataloaders)
                model_saver.save(model, global_step)
        if global_step >= opts.num_train_steps:
            break
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'Step {global_step}: start validation')
        validate(model, val_dataloaders)
        model_saver.save(model, global_step)

def validate(model, val_dataloaders):
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        if task.startswith('mlm_wwm'):
            val_log = validate_mlm(model, loader, task)
        elif task.startswith('mlm'):
            val_log = validate_mlm(model, loader, task)
        elif task.startswith('melm_wwm'):
            val_log = validate_melm(model, loader, task)
        elif task.startswith('melm'):
            val_log = validate_melm(model, loader, task)
        elif task.startswith('mrfr') and args.use_visual:
            val_log = validate_mrfr(model, loader, task)
        elif task.startswith('merfr') and args.use_visual:
            val_log = validate_mrfr(model, loader, task)
        elif task.startswith('mspanrfr') and args.use_visual:
            val_log = validate_mrfr(model, loader, task)
        elif task.startswith('monespanrfr') and args.use_visual:
            val_log = validate_mrfr(model, loader, task)
        elif task.startswith('mrc') and args.use_visual:
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('merc') and args.use_visual:
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('mspanrc') and args.use_visual:
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('monespanrc') and args.use_visual:
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('vfom') and args.use_visual:
            val_log = validate_vfom(model, loader, task)
        elif task.startswith('sfom') and args.use_speech:
            val_log = validate_sfom(model, loader, task)
        elif task.startswith('msrfr') and args.use_speech:
            val_log = validate_msrfr(model, loader, task)
        elif task.startswith('mspansrfr') and args.use_speech:
            val_log = validate_msrfr(model, loader, task)
        elif task.startswith('monespansrfr') and args.use_speech:
            val_log = validate_msrfr(model, loader, task)
        elif task.startswith('emocls'):
            val_log = validate_emocls(model, loader, emocls_type=args.emocls_type)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader, task)
        elif task.startswith('eitm'):
            val_log = validate_itm(model, loader, task)
        elif task.startswith('onemodalnegitm'):
            val_log = validate_itm(model, loader, task)
        elif task.startswith('vtm') and args.use_visual:
            val_log = validate_itm(model, loader, task)
        elif task.startswith('stm') and args.use_speech:
            val_log = validate_itm(model, loader, task)
        elif task.startswith('emolare'):
            LOGGER.info("start running EmoLare validation...")
            val_log = validate_emolare(model, loader)
        elif task.startswith('promptmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('cmpromptmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('flexpromptmissmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('cmflexpromptmissmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('flexpromptmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('cmflexpromptmask'):
            val_log = validate_prompt_mask(model, loader)
        elif task.startswith('promptnsp'):
            val_log = validate_prompt_nsp(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()

@torch.no_grad()
def validate_mlm(model, val_loader, task='mlm'):
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task=task, compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_emolare(model, val_loader):
    LOGGER.info("start running EmoLare validation...")
    val_loss = 0
    n_correct = 0
    # for pos
    val_loss_pos = 0
    n_correct_pos = 0
    # for word senti
    val_loss_wsenti = 0
    n_correct_wsenti = 0
    # for utt senti
    val_loss_usenti = 0
    n_correct_usenti = 0
    n_word = 0
    n_pos = 0
    n_wsenti = 0
    n_utt = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores, pos_scores, wsenti_scores, usenti_scores, \
                    lm_loss, pos_loss, wsenti_loss, usenti_loss = model(batch, task='emolare', compute_loss=False)
        # for token prediction
        txt_labels = batch['txt_labels']
        batch_size = txt_labels.size(0)
        # 做一次筛选
        # LOGGER.info('[Debug] the lm loss {} scores is {} and labels {}'.format(lm_loss.shape, scores.shape, txt_labels.shape))
        masked_txt_labels = txt_labels[txt_labels != -1]
        masked_lm_loss = lm_loss.view(batch_size, -1)[txt_labels != -1]
        masked_scores = scores[txt_labels != -1]
        # LOGGER.info('[Debug] the masked lm loss {} masked scores is {} and masked labels {}'.format(masked_lm_loss.shape, masked_scores.shape, masked_txt_labels.shape))
        val_loss += masked_lm_loss.mean().item()
        n_correct += (masked_scores.max(dim=-1)[1] == masked_txt_labels).sum().item()
        # for token pos prediction
        pos_labels = batch['txt_pos_labels']
        masked_pos_labels = pos_labels[pos_labels != -1]
        masked_pos_loss = pos_loss.view(batch_size, -1)[pos_labels != -1]
        masked_pos_scores = pos_scores[pos_labels != -1]
        val_loss_pos += masked_pos_loss.mean().item()
        n_correct_pos += (masked_pos_scores.max(dim=-1)[1] == masked_pos_labels).sum().item()
        # for token senti prediction
        txt_senti_labels = batch['txt_senti_labels']
        masked_txt_senti_labels = txt_senti_labels[txt_senti_labels != -1]
        masked_wsenti_loss = wsenti_loss.view(batch_size, -1)[txt_senti_labels != -1]
        masked_wsenti_scores = wsenti_scores[txt_senti_labels != -1]
        val_loss_wsenti += masked_wsenti_loss.mean().item()
        n_correct_wsenti += (masked_wsenti_scores.max(dim=-1)[1] == masked_txt_senti_labels).sum().item()
        # for utt senti prediction, the condition of all is -1.
        txt_utt_senti_labels = batch['sentence_polarity_label']
        masked_utt_senti_labels = txt_utt_senti_labels[txt_utt_senti_labels != -1]
        if masked_utt_senti_labels.size(0) == 0:
            n_correct_usenti += 0
            val_loss_usenti += 0
            n_utt += 0
        else:
            masked_usenti_loss = usenti_loss.view(batch_size, -1)[txt_utt_senti_labels != -1]
            masked_usenti_scores = usenti_scores[txt_utt_senti_labels != -1]
            val_loss_usenti += masked_usenti_loss.mean().item()
            # print(f'[Debug]masked_usenti_scores {masked_usenti_scores.shape} masked_txt_senti_labels {masked_txt_senti_labels.shape}')
            n_correct_usenti += (masked_usenti_scores.max(dim=-1)[1] == masked_utt_senti_labels).sum().item()
            # print(f'[Debug] cur batch n_correct_usenti {(masked_usenti_scores.max(dim=-1)[1] == masked_txt_senti_labels).sum().item()} and n_utt {masked_txt_senti_labels.numel()}')
            n_utt += masked_utt_senti_labels.numel()
        n_word += masked_txt_labels.numel()
        n_pos += masked_pos_labels.numel()
        n_wsenti += masked_txt_senti_labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    # for pos 
    val_loss_pos = sum(all_gather_list(val_loss_pos))
    n_correct_pos = sum(all_gather_list(n_correct_pos))
    # for w senti 
    val_loss_wsenti = sum(all_gather_list(val_loss_wsenti))
    n_correct_wsenti = sum(all_gather_list(n_correct_wsenti))
    # for u senti
    val_loss_usenti = sum(all_gather_list(val_loss_usenti))
    n_correct_usenti = sum(all_gather_list(n_correct_usenti))
    n_word = sum(all_gather_list(n_word))
    n_pos = sum(all_gather_list(n_pos))
    n_wsenti = sum(all_gather_list(n_wsenti))
    n_utt = sum(all_gather_list(n_utt))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_loss_pos /= n_pos
    acc_pos = n_correct_pos / n_pos
    val_loss_wsenti  /= n_wsenti
    acc_wsenti = n_correct_wsenti / n_wsenti
    if n_utt == 0:
        val_loss_usenti, acc_usenti =0, 0
    else:
        val_loss_usenti /= n_utt
        acc_usenti =  n_correct_usenti / n_utt
    val_log = {'loss': val_loss,
               'acc': acc,
               'loss_pos': val_loss_pos,
               'acc_pos': acc_pos,
               'loss_wsenti': val_loss_wsenti,
               'acc_wsenti': acc_wsenti,
               'loss_usenti': val_loss_usenti,
               'acc_usenti': acc_usenti,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}, acc_pos: {acc_pos*100:.2f}, acc_wsenti: {acc_wsenti*100:.2f}, acc_usenti:{acc_usenti*100:.2f} ")
    return val_log

@torch.no_grad()
def validate_melm(model, val_loader, task='melm'):
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    # Jinming: add emo-cls loss for melm multi-task
    use_melm_emo = False
    n_correct_emo = 0
    val_loss_emo = 0

    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task=task, compute_loss=False)
        # Jinming: add emo-scores loss for melm multi-task
        if len(scores) == 2:
            use_melm_emo = True
            scores, emo_scores = scores
        else:
            emo_scores = None
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
        # Jinming: add emo acc for melm multi-task
        if emo_scores is not None:
            emo_labels = batch['txt_emo_labels']
            emo_labels = emo_labels[emo_labels != -1]
            loss1 = F.cross_entropy(emo_scores, emo_labels, reduction='sum')
            val_loss_emo += loss1
            n_correct_emo += (emo_scores.max(dim=-1)[1] == emo_labels).sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_word = sum(all_gather_list(n_word))
    tot_time = time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss, 'acc': acc, 'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    if use_melm_emo:
        # val_loss_emo = sum(all_gather_list(val_loss_emo))
        n_correct_emo = sum(all_gather_list(n_correct_emo))
        # val_loss_emo /= n_word
        acc = n_correct_emo / n_word
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"emo acc: {acc*100:.2f}")
    return val_log

def accuracy_count(out, labels):
    outputs = out.max(dim=-1)[1]
    mask = labels != -1
    n_correct = (outputs == labels).masked_select(mask).sum().item()
    return n_correct


@torch.no_grad()
def validate_mrfr(model, val_loader, task):
    # task: mrfr or merfr
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        if task.startswith('mrfr'):
            loss = model(batch, task='mrfr', compute_loss=True)
        elif task.startswith('merfr'):
            loss = model(batch, task='merfr', compute_loss=True)
        elif task.startswith('mspanrfr'):
            loss = model(batch, task='mspanrfr', compute_loss=True)
        elif task.startswith('monespanrfr'):
            loss = model(batch, task='monespanrfr', compute_loss=True)
        else:
            LOGGER.info(f'[Error in valid_mrfr] Error task name {task}')
        val_loss += loss.sum().item() / IMG_DIM
        n_feat += batch['img_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_log = {'loss': val_loss,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log

@torch.no_grad()
def validate_vfom(model, val_loader, task):
    # task: visual frame order modeling
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_correct = 0
    n_frame = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='vfom', compute_loss=False)
        labels = batch['targets']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_frame += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_frame = sum(all_gather_list(n_frame))
    tot_time = time()-st
    val_loss /= n_frame
    acc = n_correct / n_frame
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_frame/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_sfom(model, val_loader, task):
    # task: speech frame order modeling
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_correct = 0
    n_frame = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='sfom', compute_loss=False)
        labels = batch['targets']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_frame += labels.numel()
    val_loss = sum(all_gather_list(val_loss))
    n_correct = sum(all_gather_list(n_correct))
    n_frame = sum(all_gather_list(n_frame))
    tot_time = time()-st
    val_loss /= n_frame
    acc = n_correct / n_frame
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_frame/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_msrfr(model, val_loader, task='msrfr'):
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        if task.startswith('msrfr'):
            loss = model(batch, task='msrfr', compute_loss=True)
        elif task.startswith('mspansrfr'):
            loss = model(batch, task='mspansrfr', compute_loss=True)
        elif task.startswith('monespansrfr'):
            loss = model(batch, task='monespansrfr', compute_loss=True)
        else:
            LOGGER.info(f'[Error in valid_mrfr] Error task name {task}')
        val_loss += loss.sum().item() / Speech_DIM
        n_feat += batch['speech_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_log = {'loss': val_loss,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"loss: {val_loss:.2f}")
    return val_log

@torch.no_grad()
def validate_mrc(model, val_loader, task):
    # task: mrfr or merfr
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        if task.startswith('mrc'):
            prediction_soft_label = model(batch, task='mrckl', compute_loss=False)     
        elif task.startswith('merc'):
            prediction_soft_label = model(batch, task='merckl', compute_loss=False)   
        elif task.startswith('mspanrc'):
            prediction_soft_label = model(batch, task='mspanrckl', compute_loss=False)   
        elif task.startswith('monespanrc'):
            prediction_soft_label = model(batch, task='monespanrckl', compute_loss=False)   
        else:
            LOGGER.info(f'[Error in valid_mrc] error task name {task}')
        # default use "kl" in task:
        prediction_soft_label = F.log_softmax(
            prediction_soft_label, dim=-1)
        label_targets = batch['label_targets']
        loss = F.kl_div(
            prediction_soft_label, label_targets, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(
            prediction_soft_label, label_targets)
        val_loss += loss.item()
        n_feat += batch['img_mask_tgt'].sum().item()
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log


@torch.no_grad()
def validate_emocls(model, val_loader, emocls_type='soft'):
    LOGGER.info("start running EmoCls validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label = model(
            batch, task='emocls', compute_loss=False)
        label_targets = batch['targets']

        if emocls_type == 'soft' or emocls_type == 'logits':
            if emocls_type == 'logits':
                # the defalult 
                label_targets = label_targets.true_divide(args.emocls_temperture)
                prediction_soft_label = prediction_soft_label.true_divide(args.emocls_temperture)
            # default use "kl" in task:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='sum')
            tot_score += compute_accuracy_for_soft_targets(
                prediction_soft_label, label_targets)
        else:
            loss = F.cross_entropy(prediction_soft_label, label_targets, reduction='sum')
            tot_score += (prediction_soft_label.max(dim=-1)[1] == label_targets).sum().item()

        val_loss += loss.item()
        n_feat += batch['input_ids'].size(0)
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_feat = sum(all_gather_list(n_feat))
    tot_time = time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct


@torch.no_grad()
def validate_itm(model, val_loader, task):
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        if task.startswith('itm'):
            scores = model(batch, task='itm', compute_loss=False)
        elif task.startswith('eitm'):
            scores = model(batch, task='eitm', compute_loss=False)
        elif task.startswith('vtm'):
            scores = model(batch, task='vtm', compute_loss=False)
        elif task.startswith('stm'):
            scores = model(batch, task='stm', compute_loss=False)
        elif task.startswith('onemodalnegitm'):
            scores = model(batch, task='onemodalnegitm', compute_loss=False)
        else:
            LOGGER.info(f'[Error in valid_itm] Error task name {task}')
        targets = batch['targets']
        loss = F.cross_entropy(scores, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        n_ex += len(targets)
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {'valid/loss': val_loss,
               'valid/acc': val_acc,
               'valid/ex_per_s': n_ex/tot_time}

    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log

@torch.no_grad()
def validate_prompt_mask(model, val_loader, task='promptmask'):
    # 计算 wa, uar, f1
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    total_preds = []
    total_labels = []
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task=task, compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        temp_preds = scores.argmax(axis=1)
        # print(temp_preds.shape, labels.shape)
        total_preds.append(temp_preds.detach().cpu().numpy())
        total_labels.append(labels.detach().cpu().numpy())
    val_loss = sum(all_gather_list(val_loss))
    tot_time = time()-st
    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    # ZJM: 2021.12.27 solved a bug of prompt evaluation
    candidate_list = [8699, 4963, 6517, 3407]
    # post process the prediction, 直接去掉还是替换为一个错误的结果？统计recall值的话，可以进行替换操作。
    for i in range(len(total_preds)):
        if total_preds[i] not in candidate_list:
            true_label = total_labels[i]
            error_list = candidate_list.copy()
            error_list.remove(true_label)
            total_preds[i] = random.sample(error_list, 1)[0]
    assert len(set(total_preds)) <= len(candidate_list) 
    assert len(set(total_labels)) <= len(candidate_list) 
    val_loss /= len(total_labels)
    # print(total_preds.shape, total_labels.shape)
    val_log = evaluation_metric(total_preds, total_labels)
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, ")
    LOGGER.info(f"[Validation] Loss: {val_loss:.2f},"
                                f"\t WA: {val_log['WA']*100:.2f},"
                                f"\t WF1: {val_log['WF1']*100:.2f},"
                                f"\t UA: {val_log['UA']*100:.2f},\n")
    return val_log

def compute_nsp_results(total_scores, total_gt_targets, total_fake_labels, total_imgs):
    # 首先按照同一个img的进行排序，然后每4个属于同一句话
    label_map = {0:'anger', 1:'happy', 2:'neutral', 3:'sad'}
    img2sub_scores = {}
    img2sub_fake_labels = {}
    img2sub_gt_targets = {}
    for index in range(len(total_imgs)):
        img_name = total_imgs[index]
        if img2sub_scores.get(img_name) is None:
            img2sub_scores[img_name] = [total_scores[index]]
            img2sub_fake_labels[img_name] = [total_fake_labels[index]]
            img2sub_gt_targets[img_name] = [total_gt_targets[index]]
        else:
            img2sub_scores[img_name] += [total_scores[index]]
            img2sub_fake_labels[img_name] += [total_fake_labels[index]]
            img2sub_gt_targets[img_name] += [total_gt_targets[index]]
    # one-score = (score-0, score-1)
    total_preds = []
    total_targets = []
    for img_name in img2sub_scores.keys():
        sub_scores = img2sub_scores[img_name]
        sub_fake_labels = img2sub_fake_labels[img_name]
        sub_gt_targets = img2sub_gt_targets[img_name]
        sum_probs = []
        assert len(sub_scores) == len(sub_gt_targets) == 4
        for j in range(len(sub_scores)):
            gt_target = sub_gt_targets[j]
            sum_probs.append(sub_scores[j][1])
        # print(sub_scores, sub_gt_targets, sum_probs)
        total_targets.append(gt_target)
        total_preds.append(sub_fake_labels[np.argmax(sum_probs)])
    assert len(total_preds) == len(total_targets)
    val_log = evaluation_metric(total_preds, total_targets)
    return val_log

@torch.no_grad()
def validate_prompt_nsp(model, val_loader, task='promptnsp'):
    # 计算情感识别的 wa, uar, f1 
    LOGGER.info(f"start running {task} validation...")
    val_loss = 0
    tot_score = 0
    total_imgs = []
    total_scores = []
    total_fake_labels = []
    total_gt_targets = []
    st = time()
    for i, batch in enumerate(val_loader):
        total_imgs.append(batch['img_fnames'])
        total_fake_labels.append(batch['fake_labels'].detach().cpu().numpy())
        logits = model(batch, task=task, compute_loss=False)
        scores =  torch.softmax(logits, dim=1)
        targets = batch['targets']
        # print(scores, targets)
        loss = F.cross_entropy(logits, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += (scores.max(dim=-1)[1] == targets).sum().item()
        gt_targets = batch['gt_targets']
        total_scores.append(scores.detach().cpu().numpy())
        total_gt_targets.append(gt_targets.detach().cpu().numpy())
    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    total_scores = np.concatenate(total_scores)
    total_fake_labels = np.concatenate(total_fake_labels)
    total_gt_targets = np.concatenate(total_gt_targets)
    total_imgs = np.concatenate(total_imgs)
    assert len(total_scores) == len(total_gt_targets) == len(total_imgs) == 4 * len(set(total_imgs)) == len(total_fake_labels)
    tot_time = time()-st
    val_loss /= len(total_scores)
    val_acc = tot_score / len(total_scores)
    tot_time = time()-st
    # print(total_preds.shape, total_labels.shape)
    val_log = compute_nsp_results(total_scores, total_gt_targets, total_fake_labels, total_imgs)
    LOGGER.info(f"validation finished in {int(tot_time)} seconds,  and NSP acc {val_acc}")
    LOGGER.info(f"[Validation] Loss: {val_loss:.2f},"
                                f"\t WA: {val_log['WA']*100:.2f},"
                                f"\t WF1: {val_log['WF1']*100:.2f},"
                                f"\t UA: {val_log['UA']*100:.2f},\n")
    return val_log



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--config', required=True, help='JSON config files')
    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")
    parser.add_argument("--checkpoint_step", default=0, type=int,
                        help="which step continue to train")
    parser.add_argument("--is_reinit_lr", action='store_true',
                        help="Note: use with warmup_steps=0, when continue train and lr is reinit or not!")
    parser.add_argument("--lr_sched_type", default='linear_decay',
                        help="[fixed, linear]")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    parser.add_argument('--cvNo', type=int, required=True, default=0,
                        help='which cross-valiation folder, \
                        if cvNo=0, then donot use th cross-validation')
    parser.add_argument('--melm_prob', default=0.5, type=float,
                        help='probability to mask in MELM training')
    # traditional task
    parser.add_argument('--mlm_prob', default=0.15, type=float,
                        help='probability to mask in MLM or WWM-MLM training')
    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--msrm_prob', default=0.15, type=float,
                        help='probability to mask in MSRM training(for speech)')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--itm_neg_img_prob', default=0.5, type=float,
                        help='probability to instead img modality as negative examples'
                             'in ITM training')
    parser.add_argument('--itm_neg_samples', default=150, type=int,
                        help='batch negative examples ITM training')
    parser.add_argument('--itm_ot_lambda', default=0.0, type=float,
                        help='weight of OT (optimal transport) loss (WRA)')
    parser.add_argument('--vfom_random_prob', default=0.15, type=float,
                        help='probability to mask in vfom_random_prob visual training')
    parser.add_argument('--sfom_random_prob', default=0.15, type=float,
                        help='probability to mask in vfom_random_prob speech training')

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')
    parser.add_argument('--IMG_DIM', type=int, default=342,
                        help='visual features as transformer input')
    parser.add_argument('--Speech_DIM', type=int, default=130,
                        help='speech features as transformer input')
    parser.add_argument('--speech_conf_th', type=float, default=1.0,
                        help='threshold for dynamic speech frames boxes')
    parser.add_argument('--max_frames', type=int, default=360,
                        help='max number of speech frames')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='min number of speech frames')

    parser.add_argument('--mixed_ratios', action='store_true', help='use mixed ratios for visual and speech masking')
    parser.add_argument('--mask_speech_consecutive', type=int, default=3,
                        help='span number of speech frames')
    parser.add_argument('--mask_speech_len_ratio', type=float, default=0.15,
                        help='span number of speech frames ratio')
    parser.add_argument('--mask_visual_consecutive', type=int, default=3,
                        help='span number of visual frames')
    parser.add_argument('--mask_visual_len_ratio', type=float, default=0.2,
                        help='span number of visual frames ratio')
    # use modality branch
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')
    parser.add_argument("--emocls_type", default='soft', type=str, help='soft, hard, logits(means logits/temp)')
    parser.add_argument("--emocls_temperture", default=2.0, type=float, help='default is 2.0')
    parser.add_argument("--emolare_LStask_ratio", default=2.0, type=float, help='default is 0.2 LS and 0.8 EF, we can choice 0.2 0.4 0.6 0.8 and so on')
    parser.add_argument("--use_total_eitm", action='store_true',  help='use random sample postive(same emo) and negative(diff emo) samples')

    ## prompt 
    parser.add_argument("--prompt_type", default='iam', type=str, help='iam, itwas')

    # training parameters
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training. ")
    parser.add_argument("--val_batch_size", default=32, type=int,
                        help="Total batch size for validation. ")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")
    parser.add_argument("--frozen_en_layers", default=0, type=int, 
                help="frozen how many layers of the pretrained model")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    args = parse_with_config(parser)

    if args.cvNo > 0:
        print('[Info] For Cross-Validation and redefine the train_datasets and val_datasets')
        for i in range(len(args.train_datasets)):
            for j in range(len(args.train_datasets[i]['db'])):
                args.train_datasets[i]['db'][j] = args.train_datasets[i]['db'][j].format(args.cvNo)
        for i in range(len(args.val_datasets)):
            for j in range(len(args.val_datasets[i]['db'])):
                args.val_datasets[i]['db'][j] = args.val_datasets[i]['db'][j].format(args.cvNo)

    if not exists(args.output_dir):
        print('[Info] the output dir {}'.format(args.output_dir))
        os.makedirs(args.output_dir)
    print('[Debug] number works {}'.format(args.n_workers))
    IMG_DIM = args.IMG_DIM
    Speech_DIM = args.Speech_DIM
    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)