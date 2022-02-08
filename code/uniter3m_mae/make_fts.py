"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

根据输入的信息，获取对应的模态的输出特征.
"""

import os, argparse
import  json
import numpy as np
import h5py
import random
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from horovod import torch as hvd
from code.uniter3m.data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, EmoClsDataset, emocls_collate)
from code.uniter3m.model.infer import UniterForExtracting, extracting
from code.uniter3m.utils.logger import LOGGER


def check(input_ids, text_mask, img_feat, visual_mask, speech_feat, speech_mask, attn_masks, frame_name):
    print('-------------------------------------------------')
    print('input_ids:', input_ids.shape, input_ids)
    print('text mask:', text_mask)
    print('-------------------------------------------------')
    print('img_feat:', img_feat.shape)
    print('visual mask:', visual_mask)
    print('-------------------------------------------------')
    print('speech_feat:', speech_feat.shape)
    print('speech mask:', speech_mask)
    print('-------------------------------------------------')
    print('attn_masks:', attn_masks.shape, attn_masks)
    print('frame_name:', frame_name)
    print('-------------------------------------------------')
    # input()

def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits extracting features: {}".format(
                    device, n_gpu, hvd.rank(), False))
    
    all_datasets = []

    data_path = '/data7/emobert/exp'
    corpus_name = 'MSP'
    cvNo = 1
    
    for setname in ['trn', 'val', 'tst']:
        text_db = f"{data_path}/evaluation/{corpus_name}/txt_db/{cvNo}/{setname}_emowords_sentiword.db"
        img_db = f"{data_path}/evaluation/{corpus_name}/feature/denseface_openface_msp_mean_std_torch/img_db/fc/feat_th0.0_max36_min10"
        # speech_db = f"{data_path}/evaluation/{corpus_name}/feature/wav2vec_db_3mean/feat_th1.0_max64_min10"
        speech_db = f"{data_path}/evaluation/{corpus_name}/feature/norm_comparE_db_5mean/feat_th1.0_max64_min10"

        # load image db
        img_db = DetectFeatLmdb(img_db,
                            opts.conf_th, opts.max_bb,
                            opts.min_bb, opts.compressed_db)

        # load speech db
        speech_db = DetectFeatLmdb(speech_db,
                            opts.conf_th, opts.max_frames,
                            opts.min_frames, opts.compressed_db)

        # load text db
        txt_db = TxtTokLmdb(text_db, -1)
        # load the dataset
        sub_dataset = EmoClsDataset(txt_db, img_db, speech_db, use_text=True)
        all_datasets.append(sub_dataset)
    
    infer_dataset = ConcatDataset(all_datasets)
    ## visual
    # save_path = os.path.join(opts.output_dir, f'denseface.h5')
    # h5f = h5py.File(save_path, 'w')
    # for sample in tqdm(infer_dataset):
    #     input_ids, img_feat, speech_feat, attn_masks, target, \
    #         frame_name, input_pos_ids, input_senti_ids, sentence_polarity_ids = sample
    #     h5f[frame_name[:-4]] = img_feat
    # print('Saved in:', save_path)
     
    ## speech
    save_path = os.path.join(opts.output_dir, f'comparE_normed.h5')
    h5f = h5py.File(save_path, 'w')
    for sample in tqdm(infer_dataset):
        input_ids, img_feat, speech_feat, attn_masks, target, \
            frame_name, input_pos_ids, input_senti_ids, sentence_polarity_ids = sample
        h5f[frame_name[:-4]] = speech_feat
    print('Saved in:', save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the inference results will be "
             "written.")

    # optional parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')
    parser.add_argument('--max_frames', type=int, default=360,
                        help='max number of speech frames')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='min number of speech frames')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='min number of speech frames')
    
    parser.add_argument('--IMG_DIM', type=int, default=342,
                        help='visual features as transformer input')
    parser.add_argument('--Speech_DIM', type=int, default=768,
                        help='speech features as transformer input')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    main(args)