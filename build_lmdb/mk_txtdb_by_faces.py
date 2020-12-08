"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess json raw input to lmdb
"""
import argparse
from collections import defaultdict
import json
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

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


def process_jsonl(jsonf, db, toker, dataset_name="", split="", filter_path=None, num_samples=0):
    '''
    {
        "segmentId": [
            "the white vase is holding various purple flowers .",
            "purple flowers in a white vase with handles .",
            "a white vase with purple flowers inside on a table .",
            "the cream colored vase holds dainty purple flowers .",
            "a white vase filled with purple flowers sitting on top of a table ."
        ]
    }
    '''
    if filter_path is not None:
        filter_dict = json.load(open(filter_path))
        print('filter_dict has {} imgs'.format(len(filter_dict)))
    else:
        filter_dict = None
    id2len = {}
    txt2img = {}  # not sure if useful
    img2txt = defaultdict(list)
    contents = json.load(open(jsonf))
    count_img = 0
    _id = 0
    for img_file_path, value in tqdm(contents.items(), desc='building txtdb', total=len(contents.keys())):
        img_fname = img_file_path.lower().split('/')[-1].split('.')[0] + '.npz'
        if filter_dict is not None and filter_dict.get(img_fname) is None:
            continue
        for sent in value:
            example = {}
            input_ids = bert_tokenize(toker, sent)
            tokens = bert_id2token(toker, input_ids)
            txt2img[_id] = img_fname
            img2txt[img_fname].append(str(_id))
            id2len[_id] = len(input_ids)
            example['id'] = str(_id)
            example['dataset'] = dataset_name
            example['split'] = split
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

    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        id2lens, txt2img, img2txt = process_jsonl(opts.input, db, toker, dataset_name=opts.dataset_name, \
                                split=opts.split, filter_path=opts.filter_path, num_samples=opts.num_samples)
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
    parser.add_argument('--num_samples', type=int, default=0,
                        help='sample number samples from input JSON as a test set')
    parser.add_argument('--toker', default='bert-base-uncased',
                        help='which BERT tokenizer to used')
    parser.add_argument('--dataset_name', default='movie110_v1',
                        help='which dataset to be processed')
    args = parser.parse_args()
    main(args)

'''
根据人脸特征数据来构建文本数据，一个文本对应一个图片
python mk_txtdb_by_faces.py --input /data7/emobert/img_db/movie110_v1/nbb_th0.2_max100_min10.json \
                --output /data7/emobert/img_db/movie110_v1/txt_db/movie110_v1_trn.db \
                --toker bert-base-uncased  --dataset_name movie110_v1
'''