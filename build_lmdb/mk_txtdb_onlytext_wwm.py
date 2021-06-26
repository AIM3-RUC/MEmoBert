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
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
from code.uniter.data.data import open_lmdb
from preprocess.tools.get_emo_words import EmoSentiWordLexicon


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
    
def get_emo_words(emolex, input_ids, tokens):
    '''
    按word进行处理
    word2emo: {token:label, token2:label}
    return: emoword and it's category
    '''
    assert len(input_ids) == len(tokens)
    new_input_ids, new_tokens = [], []
    for i in range(len(input_ids)):
        new_input_ids.extend(input_ids[i])
        new_tokens.extend(tokens[i])
    assert len(new_input_ids) == len(new_tokens)
    emo_input_ids = []
    emo_input_ids_labels = []
    word2emo = emolex.score(new_tokens)
    for i in range(len(new_tokens)):
        if word2emo[new_tokens[i]] > 0:
            emo_input_ids.append(new_input_ids[i])
            emo_input_ids_labels.append(word2emo[new_tokens[i]])
    return emo_input_ids, emo_input_ids_labels

def process_tsv(tsv_file, db, toker, max_tokens, dataset_name, num_samples=1000000, use_emo_label=False, use_emo_dim=4, emolex=None):
    '''
    emolabel, sentence
    '''
    id2len = {}
    txt2img = {}  # not sure if useful
    img2txt = defaultdict(list)
    all_data = read_csv(tsv_file, delimiter=',', skip_rows=1)
    _id = 0
    count = 0
    segmentIds = list(range(len(all_data)))
    for segmentId in tqdm(segmentIds, total=len(segmentIds)):
        img_fname = dataset_name +  '_' + str(segmentId) + '.npz'
        emo, sent = all_data[segmentId][0], all_data[segmentId][1]
        example = {}
        input_ids = bert_tokenize(toker, sent)
        if len(input_ids) > max_tokens:
            print('[Debug] inputs len {}'.format(len(input_ids)))
            input_ids = input_ids[:max_tokens]
        if len(input_ids) <= 4:
            # print(f'[Debug] {segmentId} inputs len {len(input_ids)}')
            continue
        if count >= num_samples:
            continue
        tokens = bert_id2token(toker, input_ids)
        if emolex is not None:
            emo_input_ids, emo_input_ids_labels = get_emo_words(emolex, input_ids, tokens)
            if len(emo_input_ids) == 0:
                # print(emo_input_ids, emo_input_ids_labels)
                continue
            else:
                example['emo_input_ids'] = emo_input_ids
                example['emo_input_ids_labels'] = emo_input_ids_labels
        count += 1
        txt2img[_id] = img_fname
        img2txt[img_fname].append(str(_id))
        # modified
        id2len[_id] = sum([len(word_input_ids) for word_input_ids in input_ids])
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

    if opts.select_emowords:
        print("*********** Use Emo Words to select data ************")
        bert_vocab_filepath = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
        word2score_path = '/data2/zjm/tools/EmoLexicons/sentiword2score.pkl'
        emolex = EmoSentiWordLexicon(word2score_path, bert_vocab_filepath)
    else:
        emolex = None

    if not os.path.exists(opts.output):
        os.makedirs(opts.output)
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        id2lens, txt2img, img2txt = process_tsv(opts.input, db, toker, dataset_name=opts.dataset_name, max_tokens=opts.max_text_tokens, \
                                use_emo_label=opts.use_emo_label, use_emo_dim=opts.use_emo_dim, emolex=emolex)
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
    parser.add_argument('--select_emowords',  action='store_true',
                        help='use the utts that exits emo words')
    args = parser.parse_args()
    main(args)