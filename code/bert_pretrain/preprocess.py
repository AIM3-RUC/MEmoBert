'''
将三个version的可用的文本数据整理出来，分别作为训练集和测试集合。
采用原始的数据格式
'''
import numpy as np
import json
from os import write
import random 
from tqdm import tqdm

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

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines

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
# txt_json_path = '/data7/emobert/data_nomask/movies_v1/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.1_emowords_all_6000.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# txt_json_path = '/data7/emobert/data_nomask/movies_v2/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v2_th0.1_emowords_all_4000.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# txt_json_path = '/data7/emobert/data_nomask/movies_v3/ref_captions.json'
# txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v3_th0.1_emowords_all_8000.db/img2txts.json'
# sub_sentents = read_content(txt_json_path, txt_db_path)
# all_sentents.extend(sub_sentents)
# print('total val set {}'.format(len(all_sentents)))
# save_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_val.txt'
# write_file(save_path, all_sentents)

all_sentents = []
all_json_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en.txt'
train_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn100w.txt'
val_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt'
all_lines = read_file(all_json_path)
# val_lines = random.sample(all_lines, 100000)
# write_file(val_path, val_lines)
val_lines = read_file(val_path)
val_dict = {line:1 for line in val_lines}
count = 0
trn_lines = []
all_indexs = list(range(len(all_lines)))
np.random.shuffle(all_indexs)
for index in all_indexs:
    if val_dict.get(all_lines[index]) is not None:
        continue
    trn_lines.append(all_lines[index])
    count += 1
    if count == 1000000:
        print('trn set is 1000000')
        break
write_file(train_path, trn_lines)
