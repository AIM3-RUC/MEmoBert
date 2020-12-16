"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER pre-training
根据输入的信息，获取对应的模态的输出特征.
Step1: 构建图文的合并的
"""

import os, argparse
from time import time
import numpy as np

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from code.uniter.data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, InferDataset, infer_collate)
from code.uniter.model.infer import UniterForExtracting, extracting
from code.uniter.utils.logger import LOGGER
from code.uniter.utils.distributed import all_gather_list
from code.uniter.utils.const import IMG_DIM

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
    img_db = DetectFeatLmdb(opts.img_db,
                                 opts.conf_th, opts.max_bb,
                                 opts.min_bb, opts.num_bb,
                                 opts.compressed_db)
    # load text db
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    # load the dataset
    infer_dataset = InferDataset(txt_db, img_db)

    # Prepare model
    checkpoint = torch.load(opts.checkpoint)
    model = UniterForExtracting.from_pretrained(
        opts.model_config, checkpoint, img_dim=IMG_DIM)

    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    infer_dataloader = DataLoader(infer_dataset, batch_size=opts.batch_size,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 shuffle=False,
                                 collate_fn=infer_collate)
    infer_dataloader = PrefetchLoader(infer_dataloader)

    txt_features, img_features = extracting_mm_fts(model, infer_dataloader)
    print('Final Feature txt {} img {}'.format(len(txt_features), len(img_features)))
    np.save(os.path.join(opts.output_dir, 'txt_ft.npy'), txt_features)
    np.save(os.path.join(opts.output_dir, 'face_ft.npy'), img_features)
    # Manually check the samples' length 
    for i in range(5):
        print('\ttxt original {} tokens fts {}'.format(txt_db.id2len[str(i)], txt_features[i].shape))
        print('\timg original {} faces fts {}'.format(img_db.name2nbb[txt_db.txt2img[str(i)]], img_features[i].shape))

@torch.no_grad()
def extracting_mm_fts(model, eval_loader):
    model.eval()
    st = time()
    LOGGER.info("start running Image/Text Retrieval evaluation ...")
    txt_features, img_features = extracting(model, eval_loader)
    tot_time = time()-st
    LOGGER.info(f"extracting finished in {int(tot_time)} seconds, ")
    return txt_features, img_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="model checkpoint binary")
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
    parser.add_argument("--batch_size", default=400, type=int,
                        help="number of samples in a batch")
    # device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    args = parser.parse_args()

    main(args)