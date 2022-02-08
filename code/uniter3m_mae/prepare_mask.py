"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

根据输入的信息，获取对应的模态的输出特征.
"""

import os, argparse
import json
import numpy as np
import h5py
import random
import torch
from tqdm import tqdm
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
    
    # load image db
    if opts.use_visual:
        img_db = DetectFeatLmdb(opts.img_db,
                                 opts.conf_th, opts.max_bb,
                                 opts.min_bb, opts.compressed_db)
    else:
        img_db = None

    # load speech db
    if opts.use_speech:
        speech_db = DetectFeatLmdb(opts.speech_db,
                                 opts.conf_th, opts.max_frames,
                                 opts.min_frames, opts.compressed_db)
    else:
        speech_db = None

    # load text db
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    # load the dataset
    infer_dataset = EmoClsDataset(txt_db, img_db, speech_db, use_text=opts.use_text)
    
    for i in range(opts.num_masks):
        save_path = os.path.join(opts.output_dir, f'mask_v{i}_A{opts.mask_ratio_speech}_V{opts.mask_ratio_visual}_T{opts.mask_ratio_text}.h5')
        h5f = h5py.File(save_path, 'w')
        for sample in tqdm(infer_dataset):
            input_ids, img_feat, speech_feat, attn_masks, target, \
                frame_name, input_pos_ids, input_senti_ids, sentence_polarity_ids = sample
            text_mask = np.ones(len(input_ids))
            for idx in random.sample(range(len(input_ids)-2), round((len(input_ids)-2) * opts.mask_ratio_text)):
                text_mask[idx+1] = 0
            visual_mask = np.ones(len(img_feat))
            for idx in random.sample(range(len(img_feat)), round(len(img_feat) * opts.mask_ratio_visual)):
                visual_mask[idx] = 0
            speech_mask = np.ones(len(speech_feat))
            for idx in random.sample(range(len(speech_feat)), round(len(speech_feat) * opts.mask_ratio_speech)):
                speech_mask[idx] = 0
            
            # check(input_ids, text_mask, img_feat, visual_mask, speech_feat, speech_mask, attn_masks, frame_name)
            # print(frame_name)
            # input()
            h5f[f'text/{frame_name}'] = text_mask
            h5f[f'visual/{frame_name}'] = visual_mask
            h5f[f'speech/{frame_name}'] = speech_mask    
        print('Saved in:', save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--speech_db", default=None, type=str,
                        help="The input train speech.")
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
    parser.add_argument("--use_text", action='store_true',  help='use speech branch')
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')

    # mask parameter
    parser.add_argument("--num_masks", type=int, default=3,  help='how many masks for validation')
    parser.add_argument("--mask_ratio_text", type=float, default=0.9,  help='the mask ratio of text sequences')
    parser.add_argument("--mask_ratio_speech", type=float, default=0.9,  help='the mask ratio of speech sequences')
    parser.add_argument("--mask_ratio_visual", type=float, default=0.9,  help='the mask ratio of visual sequences')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    main(args)