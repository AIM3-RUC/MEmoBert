'''
将三个version的可用的文本数据整理出来，分别作为训练集和测试集合。
采用原始的数据格式
export PYTHONPATH=/data7/MEmoBert
'''

import numpy as np
import json
from os import write
import random 
from tqdm import tqdm
from preprocess.FileOps import read_csv, write_csv, read_json
import sys
from transformers import AutoTokenizer

def read_content(jsonf, filter_path):
    # the valid segments
    valid_texts = []
    filter_dict = json.load(open(filter_path))
    print('filter_dict {}'.format(len(filter_dict)))
    contents = json.load(open(jsonf))
    segmentIds = list(contents.keys())
    for segmentId in tqdm(segmentIds, total=len(segmentIds)):
        # No0133_Community_S01E10_378
        value = contents[segmentId]
        # No0133_Community_S01E10_378.npz
        img_fname = segmentId + '.npz'
        if filter_dict.get(img_fname) is None:
            continue
        valid_texts.append(value[0] + '\n')
    print('{} total valid sentents in {}'.format(filter_path, len(valid_texts)))
    return valid_texts

if __name__ == '__main__':
    # 构建meld的文本数据集
    setname = sys.argv[1]
    target_path = '/data7/emobert/exp/evaluation/MELD/target/{}/label.npy'.format(setname)
    int2name_path = '/data7/emobert/exp/evaluation/MELD/target/{}/int2name.npy'.format(setname)
    text_path = '/data7/emobert/exp/evaluation/MELD/refs/{}.json'.format(setname)
    save_path = '/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/{}.csv'.format(setname)

    # toker = AutoTokenizer.from_pretrained('bert-base-uncased')
    all_sents = []
    all_sents_len = []
    # ['0_0', starttime, endtiem]
    int2name = np.load(int2name_path)
    target = np.load(target_path)
    all_sents.append(['label', 'sentence1'])
    # "val/dia0_utt0": {"txt": ["Oh my God, he's lost it. He's totally lost it."], "label": 3},
    text_dict = read_json(text_path)
    for i in range(len(int2name)):
        label = target[i]
        name = int2name[i][0]
        splits = name.split('_')
        key_id = '{}/dia{}_utt{}'.format(setname, splits[0], splits[1])
        text = text_dict[key_id]['txt'][0]
        label2 = text_dict[key_id]['label']
        assert label == int(label2)
        all_sents.append([label, text])
        all_sents_len.append(len(text.split(' ')))
        ### for debug
        # input_ids = toker.encode(text)
        # sub_tokens = [toker._convert_id_to_token(id) for id in input_ids]
        # print(input_ids)
        # print(sub_tokens)
    print('{} have {} {} samples and {} words'.format(setname, len(all_sents), len(int2name), sum(all_sents_len)/len(all_sents_len))) 
    write_csv(save_path, all_sents, delimiter=',')

# all_sentents = []
# txt_json_path = '/data7/emobert/data_nomask/movies_v1/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.1_emowords_all_new_trn.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# txt_json_path = '/data7/emobert/data_nomask/movies_v2/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v2_th0.1_emowords_all_new_trn.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# txt_json_path = '/data7/emobert/data_nomask/movies_v3/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v3_th0.1_emowords_all_new_trn.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# print('total trn set {}'.format(len(all_sentents)))
# save_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_trn.txt'
# write_file(save_path, all_sentents)

# all_sentents = []
# all_json_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en.txt'
# train_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn100w.txt'
# val_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt'
# all_lines = read_file(all_json_path)
# # val_lines = random.sample(all_lines, 100000)
# # write_file(val_path, val_lines)
# val_lines = read_file(val_path)
# val_dict = {line:1 for line in val_lines}
# count = 0
# trn_lines = []
# all_indexs = list(range(len(all_lines)))
# np.random.shuffle(all_indexs)
# for index in all_indexs:
#     if val_dict.get(all_lines[index]) is not None:
#         continue
#     trn_lines.append(all_lines[index])
#     count += 1
#     if count == 1000000:
#         print('trn set is 1000000')
#         break
# write_file(train_path, trn_lines)
