"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for Image-Text Retrieval
"""
import argparse
import os
import fcntl
from os.path import exists, join
from time import time
import json

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm

from code.uniter3m.data import (PrefetchLoader, TxtTokLmdb, ImageLmdbGroup, SpeechLmdbGroup, EmoClsDataset, emocls_collate)
from code.uniter3m.model.emocls import UniterForEmoRecognition, evaluation
from code.uniter3m.optim import get_lr_sched
from code.uniter3m.optim.misc import build_optimizer
from code.uniter3m.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from code.uniter3m.utils.distributed import (all_reduce_and_rescale_tensors, broadcast_tensors)
from code.uniter3m.utils.save import ModelSaver, save_training_meta
from code.uniter3m.utils.misc import NoOp, parse_with_config, set_random_seed


def build_dataloader(dataset, collate_fn, is_train, opts):
    # 构建训练集合或者测试集合的
    batch_size = opts.train_batch_size if is_train else opts.inf_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader

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

    if hvd.rank() == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
        # store ITM predictions
        if not os.path.exists(join(opts.output_dir, 'results_val')):
            os.makedirs(join(opts.output_dir, 'results_val'))
        if not os.path.exists(join(opts.output_dir, 'results_test')):
            os.makedirs(join(opts.output_dir, 'results_test'))
        if not os.path.exists(join(opts.output_dir, 'results_train')):
            os.makedirs(join(opts.output_dir, 'results_train'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info("Loading no image_data_augmentation for validation and testing")
    if opts.use_visual:
        eval_all_img_dbs = ImageLmdbGroup(compress=opts.compressed_db)
    if opts.use_speech:
        eval_all_speech_dbs = SpeechLmdbGroup(compress=opts.compressed_db)              
    # val
    LOGGER.info(f"Loading Val Dataset {opts.val_img_db}, {opts.val_txt_db}, {opts.val_speech_db}")
    opts.val_txt_db = opts.val_txt_db.format(opts.cvNo)
    val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
    if opts.use_visual:
        val_img_db = eval_all_img_dbs[opts.val_img_db]
    else:
        val_img_db = None
    if opts.use_speech:
        val_speech_db = eval_all_speech_dbs[opts.val_speech_db]
    else:
        val_speech_db = None
    val_dataset = EmoClsDataset(val_txt_db, val_img_db, val_speech_db, use_text=opts.use_text, use_emolare=opts.use_emolare)
    val_dataloader = build_dataloader(val_dataset, emocls_collate, False, opts)
    # test
    LOGGER.info(f"Loading Test Dataset {opts.test_img_db}, {opts.test_txt_db} {opts.test_speech_db}")
    opts.test_txt_db = opts.test_txt_db.format(opts.cvNo)
    test_txt_db = TxtTokLmdb(opts.test_txt_db, -1)
    if opts.use_visual:
        test_img_db = eval_all_img_dbs[opts.test_img_db]
    else:
        test_img_db = None
    if opts.use_speech:
        test_speech_db = eval_all_speech_dbs[opts.test_speech_db]
    else:
        test_speech_db = None
    test_dataset = EmoClsDataset(test_txt_db, test_img_db, test_speech_db, use_text=opts.use_text, use_emolare=opts.use_emolare)
    test_dataloader = build_dataloader(test_dataset, emocls_collate, False, opts)

    # Prepare model
    LOGGER.info('[Info] Loading from pretrained model {}'.format(opts.checkpoint))
    checkpoint = torch.load(opts.checkpoint)
    model = UniterForEmoRecognition(opts.model_config, \
                            img_dim=IMG_DIM, speech_dim=Speech_DIM, \
                            use_visual=opts.use_visual, use_speech=opts.use_speech, \
                            cls_num=opts.cls_num, \
                            frozen_en_layers=opts.frozen_en_layers, \
                            cls_dropout=opts.cls_dropout, cls_type=opts.cls_type, \
                            use_emolare=opts.use_emolare)
    model.load_state_dict(checkpoint)
    print('Load model success')

    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)

    global_step = 0
    n_examples = 0
    best_eval_metrix = 0
    best_eval_step = 0
    steps2test_results = {}
    steps2val_results = {}

    val_log = evaluation(model, val_dataloader)
    LOGGER.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                f"\t WA: {val_log['WA']*100:.2f},"
                f"\t WF1: {val_log['WF1']*100:.2f},"
                f"\t UA: {val_log['UA']*100:.2f},\n")
    test_log = evaluation(model, test_dataloader)
    LOGGER.info(f"[Testing] Loss: {test_log['loss']:.2f},"
                f"\t WA: {test_log['WA']*100:.2f},"
                f"\t WF1: {val_log['WF1']*100:.2f},"
                f"\t UA: {test_log['UA']*100:.2f},\n")
    steps2val_results[0] = val_log
    steps2test_results[0] = test_log

    ### final use the best model tested on validation set.
    LOGGER.info('Val: Best eval steps {} found with {} {}'.format(best_eval_step, select_metrix, best_eval_metrix))
    LOGGER.info('Val: {}'.format(steps2val_results[best_eval_step]))
    LOGGER.info('Test: {}'.format(steps2test_results[best_eval_step]))
    steps2val_results['beststep'] = best_eval_step
    json.dump(steps2test_results, open(join(opts.output_dir, 'log', 'step2test_reuslts.json'),'w',encoding='utf-8'))
    json.dump(steps2val_results, open(join(opts.output_dir, 'log', 'step2val_reuslts.json'),'w',encoding='utf-8'))
    write_result_to_tsv(output_tsv, steps2test_results[best_eval_step], opts.cvNo)

def write_result_to_tsv(file_path, tst_log, cvNo):
    # 1. 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
    # 2. 如果不存在先创建一个 output_csv
    if not os.path.exists(file_path):
        open(file_path, 'w').close()  # touch output_csv
    f_in = open(file_path)
    fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
    content = f_in.readlines()
    if len(content) != 12:
        content += ['\n'] * (12-len(content))
    content[cvNo-1] = 'CV{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cvNo, tst_log['WA'], tst_log['WF1'], tst_log['UA'])
    f_out = open(file_path, 'w')
    f_out.writelines(content)
    f_out.close()
    f_in.close()                              # 释放锁

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained MLM")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model "
                             "checkpoints will be written.")
    # self-modify
    parser.add_argument('--cvNo', type=int, required=True,
                        help='which cross-valiation folder')
    parser.add_argument('--frozen_en_layers', type=int, required=True,
                        help='frozen how many layers of the pretrained model')
    parser.add_argument("--cls_dropout", default=0.3, type=float,
                        help="tune dropout regularization of final classification layer")
    parser.add_argument("--cls_type", default='vqa',
                        help="for the type of the classfier layer")
    parser.add_argument("--cls_num", default=4, type=int,
                        help="number classes of the downstream tasks")
    parser.add_argument('--postfix', required=True, default='None',
                        help='postfix for the output dir')
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
    parser.add_argument("--image_data_augmentation", default=True, type=bool)
    parser.add_argument('--speech_conf_th', type=float, default=1.0,
                        help='threshold for dynamic speech frames boxes')
    parser.add_argument('--max_frames', type=int, default=360,
                        help='max number of speech frames')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='min number of speech frames')
    parser.add_argument('--IMG_DIM', type=int, default=342,
                        help='visual features as transformer input')
    parser.add_argument('--Speech_DIM', type=int, default=130,
                        help='speech features as transformer input')

    # use modality branch
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')
    parser.add_argument("--use_text", action='store_true',  help='use text branch')
    parser.add_argument("--use_emolare", action='store_true',  help='use label aware as input of text branch')

    # training parameters
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Total batch size for training. "
                             "(batch by examples)")
    parser.add_argument("--inf_batch_size", default=128, type=int,
                        help="batch size for running inference. "
                             "(used for validation, and evaluation)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=10000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=0.25, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")
    parser.add_argument("--patience", default=5, type=int,
                        help="Early stop patience")
    parser.add_argument("--lr_sched_type", default='linear_decay',
                        help="[fixed, linear]")
    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--corpus_name',  default='iemocap', type=str,
                        help="downstream task name")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    # for cross-validation
    args.output_dir = args.output_dir + '/drop{}_frozen{}_{}_{}'.format(args.cls_dropout, args.frozen_en_layers, \
                args.cls_type, args.postfix) 
    if not exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_tsv = join(args.output_dir, 'result.tsv')

    args.output_dir = output_dir = join(args.output_dir, str(args.cvNo))
    if not exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    IMG_DIM = args.IMG_DIM
    Speech_DIM = args.Speech_DIM

    if args.corpus_name == 'meld':
        select_metrix = 'WF1'
    elif args.corpus_name == 'msp':
        select_metrix = 'UA'
    elif args.corpus_name == 'iemocap':
        select_metrix = 'UA'
    LOGGER.info(f'[INFO] Corpus {args.corpus_name} and select metrix {select_metrix}')
    LOGGER.info(f'[INFO] output {args.output_dir}')

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)