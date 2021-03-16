'''
将三个version的可用的文本数据整理出来，分别作为训练集和测试集合。
采用原始的数据格式
export PYTHONPATH=/data7/MEmoBert
'''
import requests
import base64
import time
import hashlib
import numpy as np
import json
import os
from os.path import join, exists
import sys
import random 
from tqdm import tqdm
from preprocess.FileOps import read_csv, read_file, write_csv, read_json, write_file
import sys
from transformers import AutoTokenizer

# toker = AutoTokenizer.from_pretrained('bert-base-uncased')

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

def prepare2opensub():
    all_sentents = []
    txt_json_path = '/data7/emobert/data_nomask/movies_v1/ref_captions.json'
    txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.1_emowords_all_new_trn.db/img2txts.json'
    sub_sentents = read_content(txt_json_path, txt_db_path)
    all_sentents.extend(sub_sentents)
    txt_json_path = '/data7/emobert/data_nomask/movies_v2/ref_captions.json'
    txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v2_th0.1_emowords_all_new_trn.db/img2txts.json'
    sub_sentents = read_content(txt_json_path, txt_db_path)
    all_sentents.extend(sub_sentents)
    txt_json_path = '/data7/emobert/data_nomask/movies_v3/ref_captions.json'
    txt_db_path = '/data7/MEmoBert/emobert/txt_db/movies_v3_th0.1_emowords_all_new_trn.db/img2txts.json'
    sub_sentents = read_content(txt_json_path, txt_db_path)
    all_sentents.extend(sub_sentents)
    print('total trn set {}'.format(len(all_sentents)))
    save_path = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/v1v2v3_new_trn.txt'
    write_file(save_path, all_sentents)

def gen_opensubtile():
    all_filepath = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en.txt'
    val_filepath = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_val10w.txt'
    trn_filepath = '/data7/MEmoBert/emobert/exp/pretrain/only_v1v2v3_txt/data/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn1000w.txt'
    val_lines = read_file(val_filepath)
    val_dict = {line:1 for line in val_lines}
    all_lines = read_file(all_filepath)
    trn_lines = []
    for line in all_lines:
        if val_dict.get(line) is not None:
            continue
        trn_lines.append(line)
    print('total {} val {} and trn {}'.format(len(all_lines), len(val_lines), len(trn_lines)))
    write_file(trn_filepath, trn_lines)

def prepare2meld(setname):
    target_path = '/data7/emobert/exp/evaluation/MELD/target/{}/label.npy'.format(setname)
    int2name_path = '/data7/emobert/exp/evaluation/MELD/target/{}/int2name.npy'.format(setname)
    text_path = '/data7/emobert/exp/evaluation/MELD/refs/{}.json'.format(setname)
    save_path = '/data7/emobert/exp/evaluation/MELD/bert_data/{}.csv'.format(setname)
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
    print('{} have {} {} samples and {} words'.format(setname, len(all_sents), len(int2name), sum(all_sents_len)/len(all_sents_len))) 
    write_csv(save_path, all_sents, delimiter=',')

def prepare2afew_format():
    '''
    只包含 train 和 val 的数据，暂时没有文本数据，需要自己通过语音翻译
    首先从视频里面抽取音频信息. (有一些没有字幕，只有音乐背景)
    如果是找原始的字幕文件，时间可能对不上。讯飞的识别还是蛮准的, 采用讯飞的语音识别接口
    '''
    emoList = ['Surprise', 'Happy', 'Neutral', 'Sad', 'Disgust', 'Angry', 'Fear']
    root_dir = '/data2/zjm/EmotiW2017/data/Train_AFEW/'
    count = 0 
    for emo in emoList:
        video_dir = join(root_dir, emo)
        video_names = os.listdir(video_dir)
        for video_name in video_names:
            video_path = join(video_dir, video_name)
            audip_path = join(video_dir, video_name.replace('avi', 'wav'))
            os.system('ffmpeg -i {} -map 0:1 -ar 16000 -ac 1 {}'.format(video_path, audip_path))
            count += 1
    print('there are {} audios'.format(count))

def getHeader(aue, engineType, API_KEY, APPID):
    curTime = str(int(time.time()))
    param = "{\"aue\":\"" + aue + "\"" + ",\"engine_type\":\"" + engineType + "\"}"
    print("param:{}".format(param))
    paramBase64 = str(base64.b64encode(param.encode('utf-8')), 'utf-8')
    print("x_param:{}".format(paramBase64))

    m2 = hashlib.md5()
    m2.update((API_KEY + curTime + paramBase64).encode('utf-8'))
    checkSum = m2.hexdigest()
    # print('checkSum:{}'.format(checkSum))
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    return header

def getBody(filepath):
    binfile = open(filepath, 'rb')
    data = {'audio': base64.b64encode(binfile.read())}
    return data

def prepare2afew_asr():
    URL = "http://raasr.xfyun.cn/api"
    APPID = "5c11ab33"
    API_KEY = "4393e2d8f40a108dd37f7d8994c830a2"
    aue = "raw"
    engineType = "sms16k"
    audioFilePath = r"/data2/zjm/EmotiW2017/data/Val_AFEW/Angry/005150814.wav"
    r = requests.post(URL, headers=getHeader(aue, engineType, API_KEY, APPID), 
                            data=getBody(audioFilePath))
    result_json = r.json()
    write_file('test.json', result_json)

def prepare2iemocap():
    # generate the csv and txt file
    for cvNo in range(1, 11):
        for setname in ['trn', 'val', 'tst']:
            target_path = '/data7/emobert/exp/evaluation/IEMOCAP/target/{}/{}_label.npy'.format(cvNo, setname)
            int2name_path = '/data7/emobert/exp/evaluation/IEMOCAP/target/{}/{}_int2name.npy'.format(cvNo, setname)
            text_path = '/data7/emobert/exp/evaluation/IEMOCAP/refs/{}/{}_ref.json'.format(cvNo, setname)
            save_dir = '/data7/MEmoBert/emobert/exp/evaluation/IEMOCAP/bert_data/{}'.format(cvNo)
            if not exists(save_dir):
                os.mkdir(save_dir)
            save_path = join(save_dir, '{}.csv'.format(setname))
            text_save_path = join(save_dir, '{}.txt'.format(setname))
            all_sents = []
            all_text_sents = []
            all_sents_len = []
            # ['0_0', starttime, endtiem]
            int2name = np.load(int2name_path)
            target = np.load(target_path)
            all_sents.append(['label', 'sentence1'])
            # "Ses01F_impro06_M000": { "txt": ["I'm sorry, Joy."] "label": 3},            
            text_dict = read_json(text_path)
            for i in range(len(int2name)):
                label = np.argmax(target[i])
                key_id = int2name[i][0].decode('utf8')
                text = text_dict[key_id]['txt'][0]
                label2 = text_dict[key_id]['label']
                assert label == int(label2)
                all_sents.append([label, text])
                all_sents_len.append(len(text.split(' ')))
                all_text_sents.append(text + '\n')
            print('{} have {} {} samples and {} words'.format(setname, len(all_sents), len(int2name), sum(all_sents_len)/len(all_sents_len))) 
            write_csv(save_path, all_sents, delimiter=',')
            write_file(text_save_path, all_text_sents)
            print(save_path)
            print(text_save_path)

def prepare2msp():
    for cvNo in range(1, 13):
        for setname in ['trn', 'val', 'tst']:
            target_path = '/data7/emobert/exp/evaluation/MSP-IMPROV/target/{}/{}_label.npy'.format(cvNo, setname)
            int2name_path = '/data7/emobert/exp/evaluation/MSP-IMPROV/target/{}/{}_int2name.npy'.format(cvNo, setname)
            text_path = '/data7/emobert/exp/evaluation/MSP-IMPROV/refs/{}/{}_ref.json'.format(cvNo, setname)
            save_dir = '/data7/emobert/exp/evaluation/MSP-IMPROV/bert_data/{}'.format(cvNo)
            if not exists(save_dir):
                os.mkdir(save_dir)
            save_path = join(save_dir, '{}.csv'.format(setname))
            text_save_path = join(save_dir, '{}.txt'.format(setname))
            all_sents = []
            all_text_sents = []
            all_sents_len = []
            # ['0_0', starttime, endtiem]
            int2name = np.load(int2name_path)
            target = np.load(target_path)
            all_sents.append(['label', 'sentence1'])
            # "Ses01F_impro06_M000": { "txt": ["I'm sorry, Joy."] "label": 3},            
            text_dict = read_json(text_path)
            for i in range(len(int2name)):
                label = np.argmax(target[i])
                key_id = int2name[i][0].decode('utf8')
                text = text_dict[key_id]['txt'][0]
                label2 = text_dict[key_id]['label']
                assert label == int(label2)
                all_sents.append([label, text])
                all_sents_len.append(len(text.split(' ')))
                all_text_sents.append(text + '\n')
            print('{} have {} {} samples and {} words'.format(setname, len(all_sents), len(int2name), sum(all_sents_len)/len(all_sents_len))) 
            write_csv(save_path, all_sents, delimiter=',')
            write_file(text_save_path, all_text_sents)
            print(save_path)
            print(text_save_path)


def prepare2meld4mlm():
    # based on the /data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data
    # only remove the first line and first coloum
    for setname in ['train', 'val', 'test']:
        all_lines = []
        csv_path =  '/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/{}.csv'.format(setname)
        save_path = '/data7/MEmoBert/emobert/exp/evaluation/MELD/bert_data/{}.txt'.format(setname)
        instances = read_csv(csv_path, delimiter=',', skip_rows=1)
        all_lines = [ins[1]+'\n' for ins in instances]
        print('MELD setname {} and {} lines'.format(setname, len(all_lines)))
        write_file(save_path, all_lines)

if __name__ == '__main__':
    # setname = sys.argv[1]
    # gen_opensubtile()
    # prepare2afew_format()
    # prepare2afew_asr()
    # prepare2meld4mlm()
    prepare2iemocap()
