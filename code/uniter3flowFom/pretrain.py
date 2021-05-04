"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

uniterbackbone pre-training
"""
import argparse
from collections import defaultdict
import json
from os.path import join
from time import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from code.uniter3flow.data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, SpeechLmdbGroup, ConcatDatasetWithLens,
                  MlmDataset, MelmDataset, mlm_collate, melm_collate,
                  ItmDataset, itm_collate)

from code.uniter3flow.model.pretrain import MEmoBertForPretraining
from code.uniter3flow.optim import get_lr_sched, get_backbone_lr_sched
from code.uniter3flow.optim.misc import build_backbone_optimizer, build_optimizer

from code.uniter3flow.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from code.uniter3flow.utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from code.uniter3flow.utils.save import ModelSaver, save_training_meta
from code.uniter3flow.utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed, get_grad_flow
from code.uniter3flow.utils.const import IMG_LABEL_DIM, BUCKET_SIZE

def build_dataloader(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader

def build_dataloader_itm(dataset, collate_fn, is_train, opts):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = TokenBucketSamplerForItm(
        dataset, bucket_size=BUCKET_SIZE,
        batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return loader


def build_mlm_dataset(txt_db, img_db, speech_db, is_train, opts):
    ''' whether use the img/audio in create_dataloader()
    '''
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MlmDataset(t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MlmDataset(t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MlmDataset(t, i, None) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error melm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        # 在dataset里面会根据db是否为None进行相应的判断
        dataset = MlmDataset(txt_db, img_db, speech_db)
    return dataset, mlm_collate

def build_melm_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        if img_db is not None and speech_db is not None:
            datasets = [MlmDataset(opts.melm_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        elif img_db is None and speech_db is not None:
            datasets = [MlmDataset(opts.melm_prob, t, None, s) for t, s in zip(txt_db, speech_db)]
        elif img_db is not None and speech_db is None:
            datasets = [MlmDataset(opts.melm_prob, t, i, None) for t, i in zip(txt_db, img_db)]
        else:
            LOGGER.info('[Error] Error melm datasets')
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MelmDataset(opts.melm_prob, txt_db, img_db, speech_db)
    return dataset, melm_collate

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

### for build FOM dataset 
def build_fom_dataset(txt_db, img_db, speech_db, is_train, opts):
    if is_train:
        datasets = [FOMDataset(opts.fom_prob, t, i, s) for t, i, s in zip(txt_db, img_db, speech_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MelmDataset(opts.fom_prob, txt_db, img_db, speech_db)

    return dataset, fom_collate

def build_som_dataset(txt_db, img_db, speech_db, is_train, opts):
    pass

def create_dataloaders(datasets, is_train, opts, data_augmentation):
    if opts.use_visual:
        LOGGER.info('[Debug] Use ImageLmdbGroup')
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb, opts.compressed_db,
                    data_augmentation=data_augmentation)
    if  opts.use_speech:
        LOGGER.info('[Debug] Use SpeechLmdbGroup')
        all_speech_dbs = SpeechLmdbGroup(opts.speech_conf_th, opts.max_frames, opts.min_frames,
                                       opts.compressed_db)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            if dset.get('img') is not None and opts.use_visual:
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
                assert len(dset['db']) == len(dset['img']) == 1
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
                LOGGER.info(f"Loading {task} validation dataset {dset['db']}")
                txt_db = TxtTokLmdb(dset['db'][0], -1)

            if task.startswith('mlm'):
                dataset = build_mlm_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('melm'):
                dataset = build_melm_dataset(txt_db, img_db,speech_db,  is_train, opts)
            elif task.startswith('fom'):
                dataset = build_fom_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('som'):
                dataset = build_som_dataset(txt_db, img_db, speech_db, is_train, opts)
            elif task.startswith('itm'):
                dataset = build_itm_dataset(txt_db, img_db, speech_db, is_train, opts)
            else:
                raise ValueError(f'Undefined task {task}')

            LOGGER.info(f"{len(dataset[0])*hvd.size()} samples loaded")
            if task.startswith('itm'):
                # itm handles distributed training in dset not sampler
                loader = build_dataloader_itm(*dataset, is_train, opts)
            else:
                loader = build_dataloader(*dataset, is_train, opts)
            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = PrefetchLoader(loader)
    return dataloaders

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
    LOGGER.info('[Debug] use visual branch {}'.format(opts.use_visual))
    LOGGER.info('[Debug] use speech branch {}'.format(opts.use_speech))
    LOGGER.info('[Debug] image_data_augmentation {}'.format(opts.image_data_augmentation))
    LOGGER.info('[Debug] fix_visual_encoder {}'.format(opts.fix_visual_encoder))
    LOGGER.info('[Debug] fix_text_encoder {}'.format(opts.fix_text_encoder))
    LOGGER.info('[Debug] fix_speech_encoder {}'.format(opts.fix_speech_encoder))
    LOGGER.info('[Debug] fix_cross_encoder {}'.format(opts.fix_cross_encoder))
    LOGGER.info('[Debug] use_type_embedding {}'.format(opts.use_type_embedding))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
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
        assert all(tokenizer == json.load(open(f'{db}/meta.json'))['tokenizer']
                for db in all_dbs)
    else:
        tokenizer = meta_data['bert']
        assert all(tokenizer == json.load(open(f'{db}/meta.json'))['bert']
                for db in all_dbs)

    # build data loaders, default using img-data_augmentation
    train_dataloaders = create_dataloaders(
        opts.train_datasets, True, opts, data_augmentation=opts.image_data_augmentation)
    val_dataloaders = create_dataloaders(opts.val_datasets, False, opts, 
                                            data_augmentation=False)

    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    model = MEmoBertForPretraining(opts.model_config, use_speech=opts.use_speech, use_visual=opts.use_visual, \
                                        pretrained_text_checkpoint=opts.pretrained_text_checkpoint,
                                        pretrained_audio_checkpoint=opts.pretrained_audio_checkpoint,
                                        fix_text_encoder=opts.fix_text_encoder,
                                        fix_visual_encoder=opts.fix_visual_encoder,
                                        fix_speech_encoder=opts.fix_speech_encoder,
                                        fix_cross_encoder=opts.fix_cross_encoder,
                                        use_type_embedding=opts.use_type_embedding)
    if opts.checkpoint:
        LOGGER.info('[Info] Loading the whole model from pretrained model {}'.format(opts.checkpoint))
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(opts.checkpoint))
        LOGGER.info('[Debug] the missing keys {}'.format(missing_keys))
        LOGGER.info('[Debug] the unexpected keys {}'.format(unexpected_keys))
    # print('[Debug] model info {}'.format(model.state_dict().keys()))
    model.to(device)
    model.train()
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    if opts.use_backbone_optim:
        # the text branch using individual optim
        LOGGER.info('[INFO] Use 2 optimizers for backbone {} and bert model {}'.format(
            opts.optim, opts.backbone_optim))
        backbone = model.emoBert.text_encoder
        backbone_optimizer = build_backbone_optimizer(backbone, opts)   # 只包含denseface的参数
        optimizer = build_optimizer(model, opts, except_model=backbone)  # 除去denseface的参数
        model, [optimizer, backbone_optimizer]  = amp.initialize(model, [optimizer, backbone_optimizer],
                                        num_losses=len(task2scaler),
                                        enabled=opts.fp16, opt_level='O2')
    else:
        # Prepare optimizer
        LOGGER.info('[INFO] Use 1 optimizers for backbone and bert model')
        optimizer = build_optimizer(model, opts)
        model, optimizer = amp.initialize(model, optimizer,
                                            num_losses=len(task2scaler),
                                            enabled=opts.fp16, opt_level='O2')
    # LOGGER.info('[INFO] the models is \n {}'.format(model))
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    global_step = 0
    # Jinming add: restore the break checkpoint
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

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    if opts.use_backbone_optim:
        backbone_optimizer.zero_grad()
        backbone_optimizer.step()
    
    for step, (name, batch) in enumerate(meta_loader):
        n_examples[name] += batch['input_ids'].size(0)
        n_in_units[name] += (batch['text_attn_masks'] == 1).sum().item()
        # LOGGER.info('[Debug] batch size {}'.format(batch['input_ids'].size(0)))
        task = name.split('_')[0]
        loss = model(batch, task=task, compute_loss=True)
    
        n_loss_units[name] += loss.size(0)
        loss = loss.mean()  # loss is not normalized in model

        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        if opts.use_backbone_optim:
            with amp.scale_loss(loss, [optimizer, backbone_optimizer], delay_unscale=delay_unscale,
                                loss_id=task2scaler[name]) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    grads = [p.grad.data for p in model.parameters()
                            if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))
        else:
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
        
        # Jinming add, for get the grad of the last layer and the uniter first layer.
        if global_step % 200 == 0:
            layers, mean_grads = get_grad_flow(model.named_parameters())
            for layer_name, mean_grad in zip(layers, mean_grads):
                # print(layer_name)
                if layer_name == 'emoBert.text_encoder.encoder.layer.11.attention.output.LayerNorm.weight':
                    LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                    TB_LOGGER.add_scalar('backbone_text_encoder_layer11_grad', mean_grad, global_step)
                if layer_name == 'emoBert.speech_encoder.speech_encoder.layer.1.output.LayerNorm.weight':
                    LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                    TB_LOGGER.add_scalar('backbone_speech_layer1_grad', mean_grad, global_step)
                if layer_name == 'emoBert.visual_encoder.encoder.layer.1.output.LayerNorm.weight':
                    LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                    TB_LOGGER.add_scalar('backbone_visual_layer1_grad', mean_grad, global_step)
                if layer_name == 'emoBert.cross_encoder.cross_encoder.layer.0.output.LayerNorm.weight':
                    LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                    TB_LOGGER.add_scalar('cross_encoder_layer0_grad', mean_grad, global_step)

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
            if opts.use_backbone_optim:
                if opts.is_reinit_lr:
                    backbone_lr_this_step = get_backbone_lr_sched(global_step-opts.checkpoint_step, opts.lr_sched_type, opts)
                else:
                    backbone_lr_this_step = get_backbone_lr_sched(global_step, opts.lr_sched_type, opts)
                for param_group in backbone_optimizer.param_groups:
                    param_group['lr'] = backbone_lr_this_step
                TB_LOGGER.add_scalar('backbone_lr', backbone_lr_this_step, global_step)
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
                if opts.use_backbone_optim:
                    if opts.backbone_grad_norm > 0:
                        grad_norm = clip_grad_norm_(amp.master_params(backbone_optimizer),
                                                opts.backbone_grad_norm)
                        TB_LOGGER.add_scalar('backbone_grad_norm', grad_norm, global_step)
            
            optimizer.step()
            optimizer.zero_grad()
            if opts.use_backbone_optim:
                backbone_optimizer.step()
                backbone_optimizer.zero_grad()
            pbar.update(1)

            if global_step % 100 == 0:
                # monitor training throughput
                LOGGER.info(f'==============Step {global_step}===============')
                LOGGER.info('Current learning rate {}'.format(lr_this_step))
                if opts.use_backbone_optim:
                    LOGGER.info('Current backbone learning rate {}'.format(backbone_lr_this_step))
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
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('melm'):
            val_log = validate_melm(model, loader)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader)
        elif task.startswith('fom'):
            val_log = validate_fom(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        val_log = {f'{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scaler_dict(
            {f'valid_{task}/{k}': v for k, v in val_log.items()})
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time()
    for i, batch in enumerate(val_loader):
        # print('[Debug] Cur batch {} {}'.format(i, batch['txt_labels'].shape))
        scores = model(batch, task='mlm', compute_loss=False)
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
def validate_melm(model, val_loader):
    LOGGER.info("start running MELM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    # Jinming: add emo-cls loss for melm multi-task
    use_melm_emo = False
    n_correct_emo = 0
    val_loss_emo = 0

    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='melm', compute_loss=False)
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

def compute_accuracy_for_soft_targets(out, labels):
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_itm(model, val_loader):
    LOGGER.info("start running ITM validation...")
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='itm', compute_loss=False)
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
def validate_fom(model, val_loader):
    LOGGER.info("start running FOM validation...")
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='itm', compute_loss=False)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--config', required=True, type=str, help='JSON config files')
    parser.add_argument("--model_config", required=True, type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")
    parser.add_argument("--checkpoint_step", default=0, type=int,
                        help="which step continue to train")
    parser.add_argument("--is_reinit_lr", action='store_true',
                        help="Note: use with warmup_steps=0, when continue train and lr is reinit or not!")
    parser.add_argument("--lr_sched_type", default='linear_decay',
                        help="[fixed, linear_decay]")
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
    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--itm_ot_lambda', default=0.0, type=float,
                        help='weight of OT (optimal transport) loss (WRA)')
     parser.add_argument('--fom_prob', default=0.20, type=float, 
                help='probability to shuffle number tokens in FOM training')
    
    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes (-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=36,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--speech_conf_th', type=float, default=1.0,
                        help='threshold for dynamic speech frames boxes')
    parser.add_argument('--max_frames', type=int, default=360,
                        help='max number of speech frames')
    parser.add_argument('--min_frames', type=int, default=10,
                        help='min number of speech frames')
    
    # use modality branch
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')

    # backbone parameters
    parser.add_argument("--pretrained_text_checkpoint", default=None, type=str,
                                    help='the path of the pretrained text checkpoint')
    parser.add_argument("--pretrained_audio_checkpoint", default=None, type=str,
                                    help='the path of the pretrained text checkpoint')
    parser.add_argument("--image_data_augmentation", action='store_true')
    parser.add_argument("--add_cls_token", action='store_true')
    parser.add_argument("--use_backbone_optim", action='store_true',
                                    help='use individual backbone optim for text-bert training')
    parser.add_argument("--backbone_optim", default='adamw', type=str,
                                    help='use backbone optim for text-bert training')
    parser.add_argument("--backbone_learning_rate", default=1e-5, type=float, help="The initial learning rate of text backbone for Adam.")
    parser.add_argument("--backbone_weight_decay", default=1e-5, type=float)
    parser.add_argument("--backbone_warmup_steps", default=0, type=int)
    parser.add_argument("--backbone_grad_norm", default=5.0, type=float)

    # for fix or update backbone
    parser.add_argument("--fix_visual_encoder", action='store_true')
    parser.add_argument("--fix_text_encoder", action='store_true')
    parser.add_argument("--fix_speech_encoder", action='store_true')
    parser.add_argument("--fix_cross_encoder", action='store_true')
    parser.add_argument("--use_type_embedding", action='store_true')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
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
        LOGGER('[Info] For Cross-Validation and redefine the train_datasets and val_datasets')
        for i in range(len(args.train_datasets)):
            args.train_datasets[i]['db'][0] = args.train_datasets[i]['db'][0].format(args.cvNo)
        for i in range(len(args.val_datasets)):
            args.val_datasets[i]['db'][0] = args.val_datasets[i]['db'][0].format(args.cvNo)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert 36 + args.max_txt_len + 2 <= 512

    main(args)
