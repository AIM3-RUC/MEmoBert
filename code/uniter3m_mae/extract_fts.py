"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

根据输入的信息，获取对应的模态的输出特征.
"""

import os, argparse
import  json
from time import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from code.uniter3m.data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, EmoClsDataset, emocls_collate)
from code.uniter3m.model.infer import UniterForExtracting, extracting
from code.uniter3m.utils.logger import LOGGER

def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits extracting features: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))
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

    # Prepare model
    if os.path.isfile(opts.checkpoint):
        LOGGER.info('Restore from checkpoint {}'.format(opts.checkpoint))
        checkpoint = torch.load(opts.checkpoint)
    else:
        LOGGER.error('Some error when restore from checkpoint {}'.format(opts.checkpoint))

    model = UniterForExtracting.from_pretrained(
        opts.model_config, checkpoint, opts.IMG_DIM, opts.Speech_DIM, opts.use_visual, opts.use_speech)

    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    infer_dataloader = DataLoader(infer_dataset, batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=emocls_collate)
    infer_dataloader = PrefetchLoader(infer_dataloader)

    cls_features, txt_features, img_features, speech_features, targets = extracting_mm_fts(model, infer_dataloader)
    LOGGER.info('Final Feature txt {} img {} speech {} target {}'.format(len(txt_features), len(img_features), len(speech_features), len(targets)))
    if len(txt_features) > 0:
        np.save(os.path.join(opts.output_dir, 'txt_ft.npy'), txt_features)
    if len(img_features) > 0:
        np.save(os.path.join(opts.output_dir, 'img_ft.npy'), img_features)
    if len(speech_features) > 0:
        np.save(os.path.join(opts.output_dir, 'speech_ft.npy'), speech_features)
    np.save(os.path.join(opts.output_dir, 'label.npy'), targets)
    # Manually check the samples' length 
    for i in range(5):
        if len(txt_features) > 0:
            print('\ttxt original {} tokens fts {}'.format(txt_db.id2len[str(i)], txt_features[i].shape))
        if len(img_features) > 0:
            print('\timg original {} faces fts {}'.format(img_db.name2nbb[txt_db.txt2img[str(i)]], img_features[i].shape))
        if len(speech_features) > 0:
            print('\tspeech original {} speech fts {}'.format(speech_db.name2nbb[txt_db.txt2img[str(i)]], speech_features[i].shape))

@torch.no_grad()
def extracting_mm_fts(model, eval_loader):
    model.eval()
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    cls_features, txt_features, img_features, speech_features, targets = extracting(model, eval_loader)
    tot_time = time()-st
    LOGGER.info(f"extracting finished in {int(tot_time)} seconds, ")
    return cls_features, txt_features, img_features, speech_features, targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--speech_db", default=None, type=str,
                        help="The input train speech.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="model checkpoint binary, filepath or dictionary")
    parser.add_argument("--model_config", default=None, type=str,
                        help="model config json")
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
    parser.add_argument("--batch_size", default=400, type=int,
                        help="number of samples in a batch")
    
    parser.add_argument('--IMG_DIM', type=int, default=342,
                        help='visual features as transformer input')
    parser.add_argument('--Speech_DIM', type=int, default=768,
                        help='speech features as transformer input')
    parser.add_argument("--use_text", action='store_true',  help='use speech branch')
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')
    # device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    main(args)