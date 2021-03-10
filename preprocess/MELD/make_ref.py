import json
import os
import numpy as np
from glob import glob
import pandas as pd

def make_ref(utt_ids, labels, txt_info, save_path):
    ans = {}
    for utt_id, label in zip(utt_ids, labels):
        ans[utt_id] = {
            'txt': [txt_info[utt_id]],
            'label': label
        }
    json.dump(ans, open(save_path, 'w', encoding='utf8'), indent=4)

def make_text(csv_path, set_name):
    df = pd.read_csv(csv_path, encoding='utf8')
    ret = {}
    labels = []
    label_index = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    for _, row in df.iterrows():
        dia_num = int(row['Dialogue_ID'])
        utt_num = int(row['Utterance_ID'])
        utt_id = f'{set_name}/dia{dia_num}_utt{utt_num}'
        text = row['Utterance']
        ret[utt_id] = text
        label = row['Emotion']
        labels.append(label_index[label])
    return ret, labels


if __name__ == '__main__':
    save_root = '/data7/MEmoBert/evaluation/MELD/refs'
    # train
    csv_path = '/data2/ljj/MELD.Raw/train_sent_emo.csv'
    txt_info, labels = make_text(csv_path, 'train')
    make_ref(list(txt_info.keys()), labels, txt_info, os.path.join(save_root, 'train.json'))
    # val
    csv_path = '/data2/ljj/MELD.Raw/dev_sent_emo.csv'
    txt_info, labels = make_text(csv_path, 'val')
    make_ref(list(txt_info.keys()), labels, txt_info, os.path.join(save_root, 'val.json'))

    # test
    csv_path = '/data2/ljj/MELD.Raw/test_sent_emo.csv'
    txt_info, labels = make_text(csv_path, 'test')
    make_ref(list(txt_info.keys()), labels, txt_info, os.path.join(save_root, 'test.json'))