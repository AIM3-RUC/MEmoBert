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
from collections import defaultdict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm

from code.uniter3m.data import (PrefetchLoader, MetaLoader, TokenBucketSampler, ConcatDatasetWithLens, 
                    TxtTokLmdb, ImageLmdbGroup, SpeechLmdbGroup, 
                    EmoClsDataset, emocls_collate)
from code.uniter3m.model.emocls import UniterForEmoRecognition, evaluation, evaluation_miss_conditions
from code.uniter3m.optim import get_lr_sched
from code.uniter3m.optim.misc import build_optimizer
from code.uniter3m.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from code.uniter3m.utils.distributed import (all_reduce_and_rescale_tensors, broadcast_tensors)
from code.uniter3m.utils.save import ModelSaver, save_training_meta
from code.uniter3m.utils.misc import NoOp, parse_with_config, set_random_seed, set_dropout
from code.uniter.utils.const import IMG_LABEL_DIM, BUCKET_SIZE

def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = opts.train_batch_size if is_train else opts.inf_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    return dataloader

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
            LOGGER.info(f'-----Task : {task}')
            if is_train:
                # cross-validation cvNo, need to rewrite the txt_db path
                LOGGER.info(f"Loading {task} train dataset {dset['db']}")
                txt_db = [TxtTokLmdb(path.format(opts.cvNo), opts.max_txt_len) for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} train dataset {dset['db']}")
                txt_db = TxtTokLmdb(dset['db'][0].format(opts.cvNo), max_txt_len=60)

            if task.startswith('emocls'):
                dataset = build_emocls_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('nolemocls'):
                dataset = build_nolemocls_dataset(txt_db, img_db, speech_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task} of dataloader')
            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders, all_img_dbs, all_speech_dbs


def build_emocls_dataset(txt_db, img_db, speech_db, is_train, opts):
    # 可以处理多种不同的情况，通过use_text来控制
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [EmoClsDataset(t, i, s, use_text=opts.use_text, use_emolare=opts.use_emolare) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [EmoClsDataset(t, None, s, use_text=opts.use_text, use_emolare=opts.use_emolare) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [EmoClsDataset(t, i, None, use_text=opts.use_text, use_emolare=opts.use_emolare) for t, i in zip(txt_db, img_db)]
        elif img_db is None and speech_db is None:
            LOGGER.info('[Debug in mlm dataset] the img and speech modality are None!')
            datasets = [EmoClsDataset(t, None, None, use_text=opts.use_text, use_emolare=opts.use_emolare) for t in txt_db]
        else:
            LOGGER.info('[Error] Error mlm_wwm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = EmoClsDataset(txt_db, img_db, speech_db, use_text=opts.use_text, use_emolare=opts.use_emolare)
    return dataset, emocls_collate

def build_nolemocls_dataset(txt_db, img_db, speech_db, is_train, opts):
    # 仅仅处理 v a va 三种不包含L的场景, 手动设置 use_text=False
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [EmoClsDataset(t, i, s, use_text=False, use_emolare=opts.use_emolare) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [EmoClsDataset(t, None, s, use_text=False, use_emolare=opts.use_emolare) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [EmoClsDataset(t, i, None, use_text=False, use_emolare=opts.use_emolare) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error mlm_wwm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = EmoClsDataset(txt_db, img_db, speech_db, use_text=False)
    return dataset, emocls_collate


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

    # 修改为之前的可以输入多个的dataset, 方便进行模态缺失的训练和测试
    meta_data = json.load(open(os.path.join(opts.val_datasets[0]['db'][0].format(opts.cvNo), 'meta.json')))
    if meta_data.get('bert') is None:
        tokenizer = meta_data['tokenizer']
    else:
        tokenizer = meta_data['bert']
    # build data loaders
    train_dataloaders, _, _ = create_dataloaders(opts.train_datasets, True, opts)
    val_dataloaders, _, _ = create_dataloaders(opts.val_datasets, False, opts)
    tst_dataloaders = val_dataloaders
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    if opts.checkpoint and os.path.exists(opts.checkpoint):
        LOGGER.info('[Info] Loading from pretrained model {}'.format(opts.checkpoint))
        checkpoint = torch.load(opts.checkpoint)
        bert_part_checkpoint = {}
        for k, v in checkpoint.items():
            if k.startswith('output'):
                pass
            else:
                bert_part_checkpoint[k] = v
    else:
        LOGGER.info('[Warning] Bert Model is from scratch!!!')
        bert_part_checkpoint = {}

    model = UniterForEmoRecognition.from_pretrained(opts.model_config, state_dict=bert_part_checkpoint, \
                            img_dim=IMG_DIM, speech_dim=Speech_DIM, \
                            use_visual=opts.use_visual, use_speech=opts.use_speech, \
                            cls_num=opts.cls_num, \
                            frozen_en_layers=opts.frozen_en_layers, \
                            cls_dropout=opts.cls_dropout, cls_type=opts.cls_type, \
                            use_emolare=opts.use_emolare)
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

    LOGGER.info(f"***** Running training on {n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # to compute training statistics
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}
    LOGGER.info(f'[Debug] {task2loss}')

    global_step = 0
    n_examples = defaultdict(int)
    best_eval_metrix = 0
    best_eval_step = 0
    steps2test_results = {}
    steps2val_results = {}

    # for early stop 
    patience = opts.patience

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    for step, (name, batch) in enumerate(meta_loader):
        n_examples[name] += batch['input_ids'].size(0)
        print('----- Current step {} task name {}'.format(step, name))
        loss = model(batch, compute_loss=True)
        loss = loss.mean()  # loss is not normalized in model
        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale,
                            loss_id=task2scaler[name]) as scaled_loss:
            scaled_loss.backward()
            if not delay_unscale:
                # gather gradients from every processes do this before unscaling to 
                # make sure every process uses the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                            if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

        task2loss[name](loss.item())
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1
            # learning rate scheduling
            lr_this_step = get_lr_sched(global_step, opts.lr_sched_type, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)
            # log loss: not gathered across GPUs for efficiency
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

            if global_step % opts.valid_steps == 0:
                LOGGER.info(
                    f"============ Evaluation Step {global_step} "
                    f"============")
                LOGGER.info("Cur learning rate {}".format(lr_this_step))
                LOGGER.info("[Train] Loss {}".format(loss))
                val_log = evaluation(model, val_dataloaders)
                TB_LOGGER.log_scaler_dict(
                    {f"valid/{k}": v for k, v in val_log.items()})
                LOGGER.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                            f"\t WA: {val_log['WA']*100:.2f},"
                            f"\t WF1: {val_log['WF1']*100:.2f},"
                            f"\t UA: {val_log['UA']*100:.2f},\n")
                test_log = evaluation(model, test_dataloaders)
                TB_LOGGER.log_scaler_dict(
                    {f"test/{k}": v for k, v in test_log.items()})
                LOGGER.info(f"[Testing] Loss: {test_log['loss']:.2f},"
                            f"\t WA: {test_log['WA']*100:.2f},"
                            f"\t WF1: {val_log['WF1']*100:.2f},"
                            f"\t UA: {test_log['UA']*100:.2f},\n")
                steps2val_results[global_step] = val_log
                steps2test_results[global_step] = test_log
                # update the current best model based on validation results
                if val_log[select_metrix] > best_eval_metrix:
                    best_eval_step = global_step
                    best_eval_metrix = val_log[select_metrix]
                    patience = opts.patience
                    LOGGER.info('Save model at {} global step'.format(global_step))
                    model_saver.save(model, global_step)
                else:
                    if global_step > opts.warmup_steps:
                        patience -= 1 
                    if opts.patience > 0 and patience <= 0:
                        break
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps or patience <= 0:
            break
    pbar.close()
    LOGGER.info(f"finished {opts.num_train_steps} steps in {time()- start} seconds!")
    ### final use the best model tested on validation set.
    LOGGER.info('Val: Best eval steps {} found with {} {}'.format(best_eval_step, select_metrix, best_eval_metrix))
    LOGGER.info('Val: {}'.format(steps2val_results[best_eval_step]))
    LOGGER.info('Test: {}'.format(steps2test_results[best_eval_step]))
    steps2val_results['beststep'] = best_eval_step
    json.dump(steps2test_results, open(join(opts.output_dir, 'log', 'step2test_reuslts.json'),'w',encoding='utf-8'))
    json.dump(steps2val_results, open(join(opts.output_dir, 'log', 'step2val_reuslts.json'),'w',encoding='utf-8'))
    write_result_to_tsv(output_tsv, steps2test_results[best_eval_step], opts.cvNo)
    # remove the others model
    clean_chekpoints(join(opts.output_dir, 'ckpt'), best_eval_step)

def clean_chekpoints(ckpt_dir, store_epoch):
    # model_step_number.pt
    for checkpoint in os.listdir(ckpt_dir):
        if not checkpoint.endswith('_{}.pt'.format(store_epoch)):
            os.remove(os.path.join(ckpt_dir, checkpoint))

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
                        help='postfix is none')
    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=120,
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
    parser.add_argument('--seed', type=int, default=1234,
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