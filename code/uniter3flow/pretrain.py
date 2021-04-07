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

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from apex import amp
from horovod import torch as hvd

from tqdm import tqdm

from code.uniter3flow.data import (TokenBucketSampler, TokenBucketSamplerForItm,
                  MetaLoader, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  MlmDataset, MelmDataset, mlm_collate, melm_collate,
                  ItmDataset, itm_collate)

from code.uniter3flow.model.pretrain import UniterForPretraining
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


def build_mlm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [MlmDataset(t, i) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MlmDataset(txt_db, img_db)

    return dataset, mlm_collate

def build_melm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [MelmDataset(opts.melm_prob, t, i) for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = MelmDataset(opts.melm_prob, txt_db, img_db)

    return dataset, melm_collate

### for build FOM dataset 
def build_fom_dataset(txt_db, img_db, is_train, opts):
    pass
    
### for build SOM dataset 
def build_som_dataset(txt_db, img_db, is_train, opts):
    pass


def build_itm_dataset(txt_db, img_db, is_train, opts):
    if is_train:
        datasets = [ItmDataset(t, i, opts.itm_neg_prob)
                    for t, i in zip(txt_db, img_db)]
        dataset = ConcatDatasetWithLens(datasets)
    else:
        dataset = ItmDataset(txt_db, img_db, opts.itm_neg_prob)
    collate_fn = itm_collate
    return dataset, collate_fn


def create_dataloaders(datasets, is_train, opts, all_img_dbs=None):
    if all_img_dbs is None:
        if is_train:
            image_data_augmentation = opts.image_data_augmentation
        else:
            image_data_augmentation = False
        all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                     opts.num_bb, opts.compressed_db, 
                                     image_data_augmentation)
    dataloaders = {}
    for dset in datasets:
        if is_train:
            assert len(dset['db']) == len(dset['img'])
            assert len(dset['tasks']) == len(dset['mix_ratio'])
            img_db = [all_img_dbs[path] for path in dset['img']]
        else:
            assert len(dset['db']) == len(dset['img']) == 1
            img_db = all_img_dbs[dset['img'][0]]

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'

            if is_train:
                LOGGER.info(f"Loading {task} train dataset "
                            f"{dset['db']}, {[img.img_dir for img in img_db]}")
                txt_db = [TxtTokLmdb(path, opts.max_txt_len)
                          for path in dset['db']]
            else:
                LOGGER.info(f"Loading {task} validation dataset, "
                            f"{dset['db']}, {img_db.img_dir}")
                txt_db = TxtTokLmdb(dset['db'][0], -1)

            if task.startswith('mlm'):
                dataset = build_mlm_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('melm'):
                dataset = build_melm_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('fom'):
                dataset = build_fom_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('som'):
                dataset = build_som_dataset(txt_db, img_db, is_train, opts)
            elif task.startswith('itm'):
                dataset = build_itm_dataset(txt_db, img_db, is_train, opts)
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
    return dataloaders, all_img_dbs


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

    # build data loaders
    train_dataloaders, all_img_dbs = create_dataloaders(
        opts.train_datasets, True, opts)
    # Jinming: for data-augmentation
    if opts.image_data_augmentation:
        LOGGER.info('[INFO] Use the augmentation and validation img is indepently as train set')
        val_dataloaders, _ = create_dataloaders(
            opts.val_datasets, False, opts)
    else:
        # if no augmentation for train, then the vlaidaiton can use same imgdb
        LOGGER.info('[INFO] Donot use the augmentation and validation img is same as train set')
        val_dataloaders, _ = create_dataloaders(
            opts.val_datasets, False, opts, all_img_dbs)
    meta_loader = MetaLoader(train_dataloaders,
                             accum_steps=opts.gradient_accumulation_steps,
                             distributed=n_gpu > 1)
    meta_loader = PrefetchLoader(meta_loader)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = UniterForPretraining.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_embedding, img_label_dim=IMG_LABEL_DIM,
        use_speech=opts.use_speech, use_visual=opts.use_visual)
    print('[Debug] model info {}'.format(model.state_dict().keys()))
    model.to(device)
    model.train()

    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}
    if opts.use_backbone_optim:
        LOGGER.info('[INFO] Use 2 optimizers for backbone {} and bert model {}'.format(
            opts.optim, opts.backbone_optim))
        backbone = model.uniter.img_embeddings.face_encoder
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
        # if global_step == 0:
        #     LOGGER.info(f'Fisrt Step for init validation')
        #     validate(model, val_dataloaders)
        # forward pass
        n_examples[name] += batch['input_ids'].size(0)
        n_in_units[name] += (batch['attn_masks'] == 1).sum().item()
        # LOGGER.info('[Debug] batch size {}'.format(batch['input_ids'].size(0)))
        task = name.split('_')[0]
        loss = model(batch, task=task, compute_loss=True)
        if task.startswith('itm'):
            # OT
            itm_loss, ot_loss = loss
            n_loss_units[name] += itm_loss.size(0)
            itm_loss = itm_loss.mean()
            loss = itm_loss
        else:
            n_loss_units[name] += loss.size(0)
            loss = loss.mean()  # loss is not normalized in model

        # backward pass
        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        if opts.use_backbone_optim:
            with amp.scale_loss(loss, [optimizer, backbone_optimizer], delay_unscale=delay_unscale,
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
                            if layer_name == 'uniter.img_embeddings.face_encoder.features.resblock4.1.bn2.weight':
                                LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                                TB_LOGGER.add_scalar('backbone_last1_layer_grad', mean_grad, global_step)
                            if layer_name == 'uniter.img_embeddings.face_encoder.features.resblock3.1.bn2.weight':
                                LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                                TB_LOGGER.add_scalar('backbone_last2_layer_grad', mean_grad, global_step)
                            if layer_name == 'uniter.img_embeddings.LayerNorm.weight':
                                LOGGER.info('[Debug] Layer {} and mean grad {}'.format(layer_name, mean_grad))
                                TB_LOGGER.add_scalar('transformer_first_layer_grad', mean_grad, global_step)
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
            # backbone learning rate 
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
        elif task.startswith('mrfr'):
            val_log = validate_mrfr(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader, task)
        elif task.startswith('itm'):
            val_log = validate_itm(model, loader)
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


@torch.no_grad()
def validate_mrfr(model, val_loader):
    LOGGER.info("start running MRFR validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    for i, batch in enumerate(val_loader):
        loss = model(batch, task='mrfr', compute_loss=True)
        val_loss += loss.sum().item() / IMG_embedding
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
def validate_mrc(model, val_loader, task):
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        prediction_soft_label = model(
            batch, task=task, compute_loss=False)
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
        scores, ot_loss = model(batch, task='itm', compute_loss=False)
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
    parser.add_argument('--IMG_DIM', type=int, default=112,
                        help='visual features as transformer input')
    parser.add_argument('--IMG_embedding', type=int, default=512,
                        help='visual feature embedding from visual encoder')
    # use modality branch
    parser.add_argument("--use_speech", action='store_true',  help='use speech branch')
    parser.add_argument("--use_visual", action='store_true',  help='use visual branch')

    # backbone parameters
    parser.add_argument("--use_backbone_optim", action='store_true',
                                    help='use individual backbone optim for training')
    parser.add_argument("--image_data_augmentation", action='store_true')
    parser.add_argument("--backbone_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate of face or audio backbone for Adam.")

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

    print('[Debug] use_backbone_optim {}'.format(args.use_backbone_optim))
    print('[Debug] image_data_augmentation {}'.format(args.image_data_augmentation))

    if args.cvNo > 0:
        print('[Info] For Cross-Validation and redefine the train_datasets and val_datasets')
        for i in range(len(args.train_datasets)):
            args.train_datasets[i]['db'][0] = args.train_datasets[i]['db'][0].format(args.cvNo)
        for i in range(len(args.val_datasets)):
            args.val_datasets[i]['db'][0] = args.val_datasets[i]['db'][0].format(args.cvNo)

    IMG_DIM = args.IMG_DIM
    IMG_embedding = args.IMG_embedding
    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
