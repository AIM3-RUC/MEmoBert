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
from preprocess.tools.get_emo_words import EmoSentiWordLexicon
from code.uniter.data.data import open_lmdb

'''
WhoWordMasking, 构建textdb的时候按word进行保存，方便就行Mask.
'''

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

def get_emo_words(emol, input_ids, tokens):
    # jinming modify: 
    # 每个词可能对应多个情感类别
    # return emo_input_ids_labels = [[], [], []]
    assert len(input_ids) == len(tokens)
    emo_input_ids = []
    emo_input_ids_labels = []
    emo_words, word2affect = emol.get_emo_words_by_tokens(tokens)
    for i in range(len(tokens)):
        if tokens[i] in emo_words:
            emo_input_ids.append(input_ids[i])
            emo_input_ids_labels.append(word2affect[tokens[i]])
    return emo_input_ids, emo_input_ids_labels

def get_emo_type_ids(emo_category_list, input_ids, emo_input_ids, emo_input_ids_labels):
    '''
    得到每个token的情感类别
    :emo_input_ids_labels [[], [], []]
    不在emo input ids 里面的为 noemoword = 0
    在 emo input ids 但是不属于 emo input ids[1:-1] 的为 affect words 中的 others = len(emo_category_list) - 1
    如果是3分类的话，那么只有 neg pos 和 neutral(others).  即此时如果属于others情感词，那么也标注为0.
    '''
    emo_type_ids = []
    for input_id in input_ids:
        if input_id in emo_input_ids:
            index = emo_input_ids.index(input_id)
            # 得到对应的情感词类别, 可能 = [negative, anx], 
            # 根据具体的 emo_category_list 确实使用哪一个, 
            emo_labels = emo_input_ids_labels[index]
            if len(emo_labels) == 1:
                emo_type = emo_labels[0]
            else:
                emo_types = []
                for e in emo_labels:
                    if e in emo_category_list:
                        emo_types.append(e)
                if len(emo_types) > 1:
                    print("[Warning] this word have multi-label {}".format(emo_types))
                emo_type  = emo_types[0]
            if  len(emo_category_list) == 3:
                # 三分类
                if emo_type in emo_category_list[1:]:
                    emo_type_ids.append(emo_category_list.index(emo_type))
                else:
                    emo_type_ids.append(0)
            else:
                # 4 分类，others类别
                if emo_type in emo_category_list[1:-1]:
                    emo_type_ids.append(emo_category_list.index(emo_type))
                else:
                    emo_type_ids.append(len(emo_category_list) - 1)
        else:
            emo_type_ids.append(0)
    return emo_type_ids

def process_jsonl(jsonf, db, toker, max_tokens=100, dataset_name="", filter_path=None, filter_path_val=None, \
                include_path=None, num_samples=0, use_emo=False, use_emo_type=None):
    '''
    {
        "segmentId": [
            "a white vase filled with purple flowers sitting on top of a table ."
        ]
    }
    '''
    if use_emo == True:
        print("*********** Use Emo Words ************")
        lexicon_dir = '/data2/zjm/tools/EmoLexicons'
        lexicon_name = 'LIWC2015Dictionary.dic'
        emol = EmoLexicon(lexicon_dir, lexicon_name, is_bert_token=False)
    else:
        emol = None

    # add for emo type ids 
    if use_emo_type == 'emo6':
        emo_category_list = ['noemoword', 'posemo', 'anx', 'anger', 'sad', 'others']
    elif use_emo_type == 'emo4':
        emo_category_list = ['noemoword', 'posemo', 'negemo', 'others']
    elif use_emo_type == 'emo3':
        emo_category_list = ['noemoword', 'posemo', 'negemo']
    else:
        emo_category_list = None
    print('**** emo_category_list {}'.format(emo_category_list))

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
    count_img = 0
    count_emo_utts = 0  # 统计包含情感词的句子
    count_emo_words = 0  # 统计包含情感词的句子中情感词的个数
    count_negemo_utts = 0  # 统计包含情感词的句子中情感词的个数
    count_posemo_utts = 0  # 统计包含情感词的句子中情感词的个数
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
            tokens = bert_id2token(toker, input_ids)
            txt2img[_id] = img_fname
            img2txt[img_fname].append(str(_id))
            id2len[_id] = len(input_ids)
            example['id'] = str(_id)
            example['dataset'] = dataset_name
            example['file_path'] = img_fname
            example['toked_caption'] = tokens
            example['input_ids'] = input_ids
            example['img_fname'] = img_fname
            if emol is not None:
                emo_input_ids, emo_input_ids_labels = get_emo_words(emol, input_ids, tokens)
                example['emo_input_ids'] = emo_input_ids  # 存储对应的情感词的id
                example['emo_labels'] = emo_input_ids_labels # 存储对应的情感类别
                if len(emo_input_ids) > 0:
                    count_emo_utts += 1
                    count_emo_words += len(emo_input_ids)
                # print(tokens)
                # print(input_ids)
                # print(emo_input_ids, emo_input_ids_labels)
                emo_type_ids = get_emo_type_ids(emo_category_list, input_ids, emo_input_ids, emo_input_ids_labels)
                example['emo_type_ids'] = emo_type_ids
                assert len(emo_type_ids) == len(input_ids)
                # print(emo_type_ids)  
            if emo_category_list.index('posemo') in emo_type_ids:
                count_posemo_utts += 1
            if emo_category_list.index('negemo') in emo_type_ids:
                count_negemo_utts += 1
            db[str(_id)] = example
            _id += 1
        count_img += 1
        if num_samples > 0 and num_samples == count_img:
            print('just sample {} as the part val set'.format(num_samples))
            break
    if count_emo_utts > 0:
        print("emotional utts {} and avg emo words {}".format(count_emo_utts, count_emo_words/count_emo_utts))
        print('include posemo word utts {} and negemo word utts {}'.format(count_posemo_utts, count_negemo_utts))
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
    parser.add_argument('--use_emo',  action='store_true',
                        help='store the emotion words and corresding labels')
    parser.add_argument('--use_emo_type',  default=None,
                        help='one of the [None, emo7, emo6, emo4, emo3]')
    args = parser.parse_args()
    main(args)