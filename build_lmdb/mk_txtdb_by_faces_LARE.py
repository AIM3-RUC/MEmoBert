"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess json raw input to lmdb
ref the process of to get more accurate word-level polarity and pos tag.
/data7/MEmoBert/code/sentilare/preprocess/prep_self.py
+ add the roberta cls label, 0 1 2 3 4 5(unknown category).
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
# for sentilare
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import StanfordPOSTagger
import numpy as np
import pandas as pd
import csv, sys, math, h5py
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer

from code.sentilare.preprocess.prep_self import process_text, load_sentinet

SentiWordNet_path = '/data2/zjm/tools/EmoLexicons/SentiWordNet_3.0.0.txt'
gloss_embedding_path = '/data7/emobert/resources/pretrained/sentilare/gloss_embedding.npy'
# Refer to https://github.com/aesuli/SentiWordNet to download SentiWordNet 3.0
sentinet, gloss_embedding, gloss_embedding_norm = load_sentinet(SentiWordNet_path, gloss_embedding_path)
print(f'sentinet {len(sentinet)} gloss_embedding {gloss_embedding.shape}')

'''
采用这个之后不能用bert的分词结果了，而必须要采用nltk的分词结果, 而论文中说采用的是 Roberta 的 5w 的词典, 
那么词如何对应到对应的输入的InputIds的呢？
作者采取了分词然后扩展的形式，如果一个word对应的某个情感，那么该word对应的所有subtokens都对应该情感。
tokens_a, pos_a, senti_a = [], [], []
for i, tok in enumerate(example.text_a_split):
    tok_list = tokenizer.tokenize(tok)
    tokens_a.extend(tok_list)
    pos_a.extend([example.text_a_pos[i]] * len(tok_list))
    senti_a.extend([example.text_a_senti[i]] * len(tok_list))
关于label的选择：
首先early-fusion的话，必须都是 hard-level lable.
late-fusion 来说可以采用 hard-level table. 也可以采用 soft-label table.
暂时统一采取 hard-level label.
'''

def bert_tokenize(tokenizer, text, text_a_pos, text_a_senti):
    '''
    text: word list 
    text_a_pos: word pos tag
    text_a_senti: word sentiment
    '''
    ids, pos_a, senti_a = [], [], []
    for i, word in enumerate(text):
        tok_list = tokenizer.tokenize(word)
        if not tok_list:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(tok_list))
        pos_a.extend([text_a_pos[i]] * len(tok_list))
        senti_a.extend([text_a_senti[i]] * len(tok_list))
    return ids, pos_a, senti_a

def bert_id2token(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = list(map(lambda x: '@@'+x if not x.isalpha() else x, tokens))
    return tokens

def get_weak_lable_list(version):
    # version, v1, v2, or v3
    all_text2img_path = f'/data7/MEmoBert/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
    all_targe_path = f'/data7/MEmoBert/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}.h5'
    all_textId2target = h5py.File(all_targe_path, 'r')
    all_text2img = json.load(open(all_text2img_path))
    assert len(all_textId2target.keys()) == len(all_text2img)
    # transfer to all imgId2target
    imgId2target = {}
    for textId in all_text2img.keys():
        img_fname = all_text2img[textId]
        target = all_textId2target[textId]
        imgId2target[img_fname] = target
    return imgId2target

def process_jsonl(jsonf, db, toker, max_tokens=100, dataset_name="", filter_path=None, filter_path_val=None, \
                include_path=None, num_samples=0, version='v1'):
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
    _id = 0
    segmentIds = list(contents.keys())
    random.shuffle(segmentIds) # 无返回值
    total_segmentIds = []
    total_text_lists = []
    for segmentId in tqdm(segmentIds, total=len(segmentIds)):
        value = contents[segmentId]
        img_fname = segmentId + '.npz'
        if filter_dict is not None and filter_dict.get(img_fname) is None:
            continue
        if filter_dict_val is not None and filter_dict_val.get(img_fname) is not None:
            continue
        if include_dict is not None and include_dict.get(img_fname) is None:
            continue
        total_segmentIds.append(img_fname)
        total_text_lists.append(value[0])
        if num_samples > 0 and num_samples == len(total_segmentIds):
            print('just sample {} as the part val set'.format(num_samples))
            break
    print('Debug the processing samples {}'.format(len(total_text_lists), len(total_segmentIds)))
    assert len(total_segmentIds) == len(total_text_lists)
    # get weak-label list
    segmentId2emoinfo = get_weak_lable_list(version)
    print('[Debug] get the total segmentId2emoinfo {}'.format(len(segmentId2emoinfo)))
    total_label_list = []
    total_probs_list = []
    total_logits_list = []
    for i, segmentId in enumerate(total_segmentIds):
        emoinfo = segmentId2emoinfo[segmentId]
        pred = emoinfo['pred'][0]
        logits = emoinfo['logits'][0]
        target = np.argmax(pred)
        total_probs_list.append(pred)
        total_logits_list.append(logits)
        total_label_list.append(target)
    assert len(total_label_list) == len(total_logits_list) == len(total_probs_list) == len(total_segmentIds)
    print('[Debug] process the total utts and get pos and word-senti')
    clean_sent_list, pos_list, senti_list, clean_label_list = process_text(total_text_lists, total_label_list, sentinet, gloss_embedding, gloss_embedding_norm)
    assert len(clean_sent_list) == len(senti_list) == len(pos_list) == len(total_segmentIds)
    assert len(clean_label_list) == len(total_label_list)
    print('[Debug] process sub-tokens')
    for i in range(len(total_text_lists)):
        text_words, pos_l, senti_l = clean_sent_list[i], pos_list[i], senti_list[i]
        input_ids, pos_a_ids, senti_a_ids = bert_tokenize(toker, text_words, pos_l, senti_l)
        # for save the info
        example = {}
        if len(input_ids) > max_tokens:
            print('[Debug] inputs len {}'.format(len(input_ids)))
            input_ids = input_ids[:max_tokens]
        tokens = bert_id2token(toker, input_ids)
        img_fname = total_segmentIds[i]
        txt2img[_id] = img_fname
        img2txt[img_fname].append(str(_id))
        id2len[_id] = len(input_ids)
        example['id'] = str(_id)
        example['dataset'] = dataset_name
        example['file_path'] = img_fname
        example['toked_caption'] = tokens
        example['input_ids'] = input_ids
        example['pos_ids'] = pos_a_ids
        example['word_senti_ids'] = senti_a_ids
        example['soft_labels'] = total_probs_list[i]
        example['logits'] = total_logits_list[i]
        example['target'] = total_label_list[i]
        # print('[Debug] img_fname {}'.format(img_fname))
        # print('[Debug] tokens {}'.format(example['toked_caption']))
        # print('[Debug] input_ids {}'.format(example['input_ids']))
        # print('[Debug] pos_ids {}'.format(example['pos_ids']))
        # print('[Debug] word_senti_ids {}'.format(example['word_senti_ids']))
        # print('[Debug] target {}'.format(example['target']))
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
        id2lens, txt2img, img2txt = process_jsonl(opts.input, db, toker, dataset_name=opts.dataset_name, \
                                filter_path=opts.filter_path, filter_path_val=opts.filter_path_val, \
                                include_path=opts.include_path, num_samples=opts.num_samples, version=opts.version)
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
    parser.add_argument('--version', required=True,
                        help='which movies version, v1, v2, v3, or v4')
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