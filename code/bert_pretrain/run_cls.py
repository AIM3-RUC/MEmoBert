#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.

# https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue_no_trainer.py

import argparse
import math
import fcntl
import os
from os.path import join, exists
import random
import numpy as np

from accelerate import Accelerator
from datasets import load_dataset
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

from code.uniter.utils.save import ModelSaver
from code.downstream.utils.logger import get_logger
from code.downstream.run_baseline import clean_chekpoints

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--cvNo", type=int, default=1)
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--validation_pred_path", type=str, default=None, help="A npy results savepath."
    )
    parser.add_argument(
        "--test_pred_path", type=str, default=None, help="A npy resutls savepath."
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--restore_pt_checkpoint",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default='42', help="A seed for reproducible training.")
    args = parser.parse_args()

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json", "tsv"], "`validation_file` should be a csv or a json file."
    if args.test_file is not None:
        extension = args.test_file.split(".")[-1]
        assert extension in ["csv", "json", "tsv"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def compute_metrics(preds, label_ids):
    assert len(preds) == len(label_ids)
    acc = accuracy_score(label_ids, preds)
    wuar = recall_score(label_ids, preds, average='weighted')
    wf1 = f1_score(label_ids, preds, average='weighted')
    uwf1 = f1_score(label_ids, preds, average='macro')
    cm = confusion_matrix(label_ids, preds)
    return {'total':len(preds), "acc": acc, "wuar": wuar, "wf1": wf1, 'uwf1': uwf1, 'cm': cm}

def main():
    args = parse_args()

    accelerator = Accelerator()
    checkpoint_dir = join(args.output_dir, 'ckpt')
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logger = get_logger(args.output_dir, 'none')
    logger.info(accelerator.state)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    logger.info("Training/evaluation parameters {}".format(args))

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    raw_datasets = load_dataset('csv', data_files=data_files)
    if args.train_file is not None:
        label_list = raw_datasets["train"].unique("label")
    else:
        label_list = raw_datasets["validation"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    #print('config', config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # 如何获取 state_dict 呢？只加载模型层，不加载分类层. 只可能出现在训练阶段
    if args.restore_pt_checkpoint is not None and args.train_file is not None:
        print('restore from {}'.format(args.restore_pt_checkpoint))
        checkpoint = torch.load(args.restore_pt_checkpoint)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if 'classifier' not in k and 'cls_layer' not in k:
                new_state_dict[k] = v
    else:
        new_state_dict = None
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        state_dict=new_state_dict)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = 'sentence1', None
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)
        result["labels"] = [label_to_id[l] for l in examples["label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names
    )
    if args.train_file is not None:
        train_dataset = processed_datasets["train"]
    
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    if not exists(join(args.output_dir, 'ckpt')):
        os.makedirs(join(args.output_dir, 'ckpt'))
    model_saver = ModelSaver(join(args.output_dir, 'ckpt'))
    # DataLoaders creation:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    if args.train_file is not None:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    if args.train_file is not None:
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, test_dataloader
        )
    else:
        model, optimizer, eval_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, eval_dataloader, test_dataloader
        )

    if args.train_file is not None:
        logger.info('In the training stage')
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs *  num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        best_eval_acc = 0
        best_eval_epoch = -1

        for epoch in range(args.num_train_epochs):
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(f'\t[LR] current {epoch} learning rate {lr}')
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
            # evaluation at every epoch
            model.eval()
            logger.info('[Evaluation]  on validation')
            eval_results = evaluation(accelerator, model, eval_dataloader)
            logger.info('\t Epoch {}: {}'.format(epoch, eval_results))
            logger.info('[Evaluation]  on testing')
            test_reuslts = evaluation(accelerator, model, test_dataloader)
            logger.info('\t Epoch {}: {}'.format(epoch, test_reuslts))
            # choose the best epoch
            if eval_results['wf1'] > best_eval_acc:
                best_eval_epoch = epoch
                best_eval_acc = eval_results['wf1']
                patience = args.patience
                # save the model
                model_saver.save(model, epoch)
                # 保存为 Hugging-face 的格式
                model.save_pretrained(join(args.output_dir, 'ckpt', 'epoch-{}'.format(epoch)))
            # for early stop
            if patience <= 0:
                break
            else:
                patience -= 1
        # print best eval result and clean the other models
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        checkpoint_path = join(args.output_dir, 'ckpt', 'model_step_{}.pt'.format(best_eval_epoch))
        if not os.path.exists(checkpoint_path):
            logger.error("Load checkpoint error, not exist such file")
            exit(0)
        ck = torch.load(checkpoint_path)
        model.load_state_dict(ck)    
        model.eval()
        logger.info('[Final Evaluation]  on validation')
        eval_results = evaluation(accelerator, model, eval_dataloader)
        logger.info('Epoch {}: {}'.format(best_eval_epoch, eval_results))
        logger.info('[Final Evaluation]  on testing')
        test_reuslts = evaluation(accelerator, model, test_dataloader)
        logger.info('Epoch {}: {}'.format(best_eval_epoch, test_reuslts))
        clean_chekpoints(join(args.output_dir, 'ckpt'), best_eval_epoch)
        output_tsv = join(os.path.split(args.output_dir)[0], 'result.csv')
        if not os.path.exists(output_tsv):
            open(output_tsv, 'w').close()  # touch output_csv
        write_result_to_tsv(output_tsv, test_reuslts, args.cvNo)
    else:
        logger.info('In the Evaluation stage')
        model.eval()
        logger.info('[Final Evaluation]  on validation')
        eval_results, eval_preds = evaluation(accelerator, model, eval_dataloader, return_pred=True)
        np.save(args.validation_pred_path, eval_preds)
        logger.info(f'total {len(eval_preds)} samples ')
        print(eval_results)
        logger.info('[Final Evaluation]  on testing')
        test_reuslts, test_preds = evaluation(accelerator, model, test_dataloader, return_pred=True)
        np.save(args.test_pred_path, test_preds)
        logger.info(f'total {len(test_preds)} samples ')
        print(test_reuslts)

def write_result_to_tsv(file_path, tst_log, cvNo):
    # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
    f_in = open(file_path)
    fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
    content = f_in.readlines()
    if len(content) != 12:
        content += ['\n'] * (12-len(content))
    content[cvNo-1] = 'CV{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cvNo, tst_log['acc'], tst_log['wf1'], tst_log['uwf1'])
    f_out = open(file_path, 'w')
    f_out.writelines(content)
    f_out.close()
    f_in.close()

def evaluation(accelerator, model, set_dataloader, return_pred=False):
    total_preds = []
    total_labels = []
    total_logits = []
    for step, batch in enumerate(set_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        temp_preds = accelerator.gather(predictions).detach().cpu().numpy()
        temp_labels = accelerator.gather(batch["labels"]).detach().cpu().numpy()
        total_preds.append(temp_preds)
        total_labels.append(temp_labels)
        temp_logits = accelerator.gather(outputs.logits).detach().cpu().numpy()
        total_logits.append(temp_logits)
    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    # set early stop and save the best models
    eval_metric = compute_metrics(total_preds, total_labels)
    if return_pred:
        total_logits = np.concatenate(total_logits)
        print('total logits shape {}'.format(total_logits.shape))
        return eval_metric, total_logits
    else:
        return eval_metric

if __name__ == "__main__":
    main()