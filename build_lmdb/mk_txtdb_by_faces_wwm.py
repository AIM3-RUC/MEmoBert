"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess json raw input to lmdb
"""
import argparse
import os
from collections import defaultdict
import json
from cytoolz import curry
from requests.api import options
from tqdm import tqdm
import random
from pytorch_pretrained_bert import BertTokenizer
from code.uniter.data.data import open_lmdb

'''
WhoWordMasking, 构建textdb的时候按word进行保存，方便就行Mask.
'''

def bert_tokenize(tokenizer, text):
    # return [word1tokens, word2tokens, word3tokens]
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.append(tokenizer.convert_tokens_to_ids(ws))
    return ids

def bert_id2token(tokenizer, ids):
    # return [word1tokens, word2tokens, word3tokens]
    tokens = []
    for words_ids in ids:
        word_tokens = tokenizer.convert_ids_to_tokens(words_ids)
        word_tokens = list(map(lambda x: '@@'+x if not x.isalpha() else x, word_tokens))
        tokens.append(word_tokens)
    return tokens

def process_jsonl(jsonf, db, toker, max_tokens=100, dataset_name="", filter_path=None, filter_path_val=None, \
                include_path=None, num_samples=0):
    '''
    {
        "segmentId": [
            "a white vase filled with purple flowers sitting on top of a table ."
        ]
    }
    '''
    if filter_path is not None:
        filter_dict = json.load(open(filter_path))
        print('filter_dict has {} imgs'.format(len(filter_dict)))
    else:
        filter_dict = None
    
    if filter_path_val is not None:
        filter_dict_val = json.load(open(filter_path_val))
        print('val filter_dict has {} imgs'.format(len(filter_dict_val)))
    else:
        filter_dict_val = None
    
    if include_path is not None:
        include_dict = json.load(open(include_path))
        print('include dict has {} imgs'.format(len(include_dict)))
    else:
        include_dict = None

    id2len = {}
    txt2img = {}  # not sure if useful
    img2txt = defaultdict(list)
    contents = json.load(open(jsonf))
    less3 = 0
    count_img = 0
    _id = 0
    segmentIds = list(contents.keys())
    random.shuffle(segmentIds) # 无返回值
    for segmentId in tqdm(segmentIds, total=len(segmentIds)):
        value = contents[segmentId]
        img_fname = segmentId + '.npz'
        if filter_dict is not None and filter_dict.get(img_fname) is None:
            continue
        if filter_dict_val is not None and filter_dict_val.get(img_fname) is not None:
            continue
        if include_dict is not None and include_dict.get(img_fname) is None:
            continue
        for sent in value:
            example = {}
            input_ids = bert_tokenize(toker, sent)
            if len(input_ids) > max_tokens:
                print('[Debug] inputs len {}'.format(len(input_ids)))
                input_ids = input_ids[:max_tokens]
            if len(input_ids) == 0:
                # print(f'[Debug] {segmentId} inputs len {len(input_ids)}')
                continue

            if len(input_ids) < 3:
                less3 += 1
            tokens = bert_id2token(toker, input_ids)
            txt2img[_id] = img_fname
            img2txt[img_fname].append(str(_id))
            id2len[_id] = sum([len(word_input_ids) for word_input_ids in input_ids])
            example['id'] = str(_id)
            example['dataset'] = dataset_name
            example['file_path'] = img_fname
            example['toked_caption'] = tokens
            example['input_ids'] = input_ids
            example['img_fname'] = img_fname
            db[str(_id)] = example
            _id += 1
        count_img += 1
        if num_samples > 0 and num_samples == count_img:
            print('just sample {} as the part val set'.format(num_samples))
            break
    print(f'words len less than 3 are {less3}')
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
        id2lens, txt2img, img2txt = process_jsonl(opts.input, db, toker, dataset_name=opts.dataset_name, \
                                filter_path=opts.filter_path, filter_path_val=opts.filter_path_val, \
                                include_path=opts.include_path, num_samples=opts.num_samples, \
                                use_emo=opts.use_emo, use_emo_type=opts.use_emo_type)
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
                        help='input JSON, like coco ref**.json')
    parser.add_argument('--output', required=True,
                        help='output dir of DB, ')
    parser.add_argument('--filter_path',
                        help='used to filter the segment Id')
    parser.add_argument('--filter_path_val', default=None,
                        help='remove the val to get the trn, ')
    parser.add_argument('--include_path', default=None,
                        help='must in this db')
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