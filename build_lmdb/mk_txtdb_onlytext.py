"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess json raw input to lmdb
"""
import argparse
import os
from collections import defaultdict
import json
from preprocess.FileOps import read_csv
from cytoolz import curry
from requests.api import options
from tqdm import tqdm
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from preprocess.tools.get_emo_words import EmoSentiWordLexicon
from code.uniter.data.data import open_lmdb

def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

def bert_id2token(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = list(map(lambda x: '@@'+x if not x.isalpha() else x, tokens))
    return tokens

def process_tsv(tsv_file, db, toker, max_tokens, dataset_name, use_emo_label=False, use_emo_dim=4):
    '''
    emolabel, sentence
    '''
    id2len = {}
    txt2img = {}  # not sure if useful
    img2txt = defaultdict(list)
    all_data = read_csv(tsv_file, delimiter=',', skip_rows=1)
    _id = 0
    segmentIds = list(range(len(all_data)))
    for segmentId in tqdm(segmentIds, total=len(segmentIds)):
        img_fname = dataset_name +  '_' + str(segmentId) + '.npz'
        emo, sent = all_data[segmentId][0], all_data[segmentId][1]
        example = {}
        input_ids = bert_tokenize(toker, sent)
        if len(input_ids) > max_tokens:
            print('[Debug] inputs len {}'.format(len(input_ids)))
            input_ids = input_ids[:max_tokens]
        if len(input_ids) <= 1:
            # print(f'[Debug] {segmentId} inputs len {len(input_ids)}')
            continue
        tokens = bert_id2token(toker, input_ids)
        txt2img[_id] = img_fname
        img2txt[img_fname].append(str(_id))
        id2len[_id] = len(input_ids)
        example['id'] = str(_id)
        example['dataset'] = dataset_name
        example['toked_caption'] = tokens
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        if use_emo_label:
            prob = np.zeros(use_emo_dim)
            example['target'] = int(emo)
            prob[int(emo)] = 1.0
            example['logits'] = prob
            example['soft_labels'] = prob
        db[str(_id)] = example
        _id += 1
    return id2len, txt2img, img2txt

def main(opts):
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))

    if not os.path.exists(opts.output):
        os.makedirs(opts.output)

    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        id2lens, txt2img, img2txt = process_tsv(opts.input, db, toker, dataset_name=opts.dataset_name, max_tokens=opts.max_text_tokens, \
                                use_emo_label=opts.use_emo_label, use_emo_dim=opts.use_emo_dim)
    print('generate id2lens {} txt2img {} img2txt {}'.format(len(id2lens), len(txt2img), len(img2txt)))
    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/img2txts.json', 'w') as f:
        json.dump(img2txt, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='input json')
    parser.add_argument('--output', required=True,
                        help='output dir of DB, ')
    parser.add_argument('--use_emo_label',  action='store_true',
                        help='use the emo labels in the input file')
    parser.add_argument('--use_emo_dim', type=int, default=4,
                        help='use the emo labels in the input file')
    parser.add_argument('--max_text_tokens', type=int, default=100,
                        help='max tokens in one sentence')
    parser.add_argument('--num_samples', type=int, default=0,
                        help='sample number samples from input JSON as a test set')
    parser.add_argument('--toker', default='bert-base-uncased',
                        help='which BERT tokenizer to used')
    parser.add_argument('--dataset_name', default='movies_v1',
                        help='which dataset to be processed')
    args = parser.parse_args()
    main(args)