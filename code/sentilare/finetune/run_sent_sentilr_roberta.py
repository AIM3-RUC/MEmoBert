# Fine-tuning SentiLARE for sentence-level sentiment classification on SST, MR, IMDB, Yelp-2/5.
# The code is modified based on run_glue.py in pytorch-transformers

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json
import fcntl

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, RobertaTokenizer, RobertaConfig, AdamW, WarmupLinearSchedule)

from modeling_sentilr_roberta import RobertaForSequenceClassification

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sent_data_utils_sentilr import convert_examples_to_features_roberta

from sent_data_utils_sentilr import sstProcessor, mrProcessor, imdbProcessor, yelp2Processor, yelp5Processor, iemocapProcessor, meldProcessor
from code.uniter3m.train_emo import clean_chekpoints, write_result_to_tsv


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

processors = {
    "iemocap": iemocapProcessor,
    "meld": meldProcessor,
    "sst": sstProcessor,
    "imdb": imdbProcessor,
    "mr": mrProcessor,
    "yelp2": yelp2Processor,
    "yelp5": yelp5Processor,
}

# SentiLARE's input embeddings include POS embedding, word-level sentiment polarity embedding,
# and sentence-level sentiment polarity embedding (which is set to be unknown during fine-tuning).
is_pos_embedding = True
is_senti_embedding = True
is_polarity_embedding = True

def set_seed(args):
    # Set the random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    # Train the model
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = 0)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    global_epoch = 0
    best_eval_metrix = 0
    best_eval_epoch = 0
    steps2val_results = {}
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        global_epoch += 1
        global_epoch_loss = 0
        global_epoch_steps = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                      'labels':         batch[-1],
                      'pos_tag_ids':    batch[3],
                      'senti_word_ids': batch[4],
                      'polarity_ids':   batch[5]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_epoch_loss += loss.mean().item()
            global_epoch_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule torch> 1.1
                model.zero_grad()
                global_step += 1

        logger.info('current training epoch: {}'.format(global_epoch))
        logger.info('learning rate: ' + str(scheduler.get_lr()[0]))
        logger.info('training loss: {}'.format(global_epoch_loss/global_epoch_steps))
        # evaluating
        val_log = evaluate(args, model, tokenizer)
        logger.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                                f"\t WA: {val_log['WA']*100:.2f},"
                                f"\t WF1: {val_log['WF1']*100:.2f},"
                                f"\t UA: {val_log['UA']*100:.2f},\n")
       
        if args.task_name == 'meld':
            select_metrix = 'WF1'
        elif args.task_name == 'msp':
            select_metrix = 'UA'
        elif args.task_name == 'iemocap':
            select_metrix = 'UA'
        else:
            select_metrix = 'WA'
        logger.info(f'task_name {args.task_name} and select_metrix {select_metrix}')
        steps2val_results[global_epoch] = val_log
        # update the current best model based on validation results
        if val_log[select_metrix] > best_eval_metrix:
            best_eval_epoch = global_epoch
            best_eval_metrix = val_log[select_metrix]
            patience = args.patience
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)        
        else:
            if args.patience > 0 and patience <= 0:
                break

    # select the best result and write to file and remove the other models
    train_iterator.close()
    logger.info('Val: Best eval steps {} found with {} {}'.format(best_eval_epoch, select_metrix, best_eval_metrix))
    logger.info('Val: {}'.format(steps2val_results[best_eval_epoch]))
    steps2val_results['bestepoch'] = best_eval_epoch
    json.dump(steps2val_results, open(os.path.join(args.output_dir, 'step2val_reuslts.json'),'w',encoding='utf-8'))
    write_result_to_tsv(args.results_path, steps2val_results[best_eval_epoch], args.cvNo)
    # remove the others model
    clean_chekpoints(args.output_dir, best_eval_epoch)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

def clean_chekpoints(ckpt_dir, store_epoch):
    # model_step_number.pt
    for checkpoint in os.listdir(ckpt_dir):
        if not checkpoint.startswith('checkpoint-{}'.format(store_epoch)):
            os.system('rm -r {}'.format(os.path.join(ckpt_dir, checkpoint)))

def write_result_to_tsv(file_path, tst_log, cvNo):
    # 1. 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
    # 2. 如果不存在先创建一个 output_csv
    if not os.path.exists(file_path):
        print('[Debug] create the {}'.format(file_path))
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

def evaluate(args, model, tokenizer, prefix=""):
    # Do validation
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    # Prepare the validation dataset
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers = 0)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # RoBERTa doesn't use segment_ids
                        'pos_tag_ids': batch[3],
                        'senti_word_ids': batch[4],
                        'polarity_ids': batch[5],
                        'labels':         batch[-1]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    acc = accuracy_score(out_label_ids, preds)
    uar = recall_score(out_label_ids, preds, average='macro')
    wf1 = f1_score(out_label_ids, preds, average='weighted')
    f1 = f1_score(out_label_ids, preds, average='macro')
    cm = confusion_matrix(out_label_ids, preds)
    model.train()
    return {'loss': eval_loss, 'WA': acc, 'WF1': wf1, 'UA': uar,  'F1': f1}


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test = False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Select the data processor according to the task name
    processor = processors[task]()
    output_mode = "classification"
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'test' if test else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_test_examples(args.data_dir) if test else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features_roberta(examples, label_list, args.max_seq_length, tokenizer, output_mode,
             cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
             cls_token=tokenizer.cls_token,
             cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
             sep_token=tokenizer.sep_token,
             sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
             pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
             pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
             mode = args.task_name)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_senti_ids = torch.tensor([f.senti_ids for f in features], dtype=torch.long)
    all_polarity_ids = torch.tensor([f.polarity_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_senti_ids, all_polarity_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--results_path", default=None, type=str, required=True,
                        help="The output results where the model predictions and checkpoints will be written.")
    parser.add_argument("--cvNo", default=1, type=int, required=True,
                        help="which cv.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--patience", default=2, type=int,
                        help="early stop.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    processor = processors[args.task_name]()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, pos_tag_embedding=is_pos_embedding, senti_embedding=is_senti_embedding, polarity_embedding=is_polarity_embedding)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    train(args, train_dataset, model, tokenizer)

if __name__ == "__main__":
    main()