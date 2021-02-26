from operator import index
import os
import json
from pytorch_pretrained_bert import BertTokenizer
import lmdb
import msgpack
from lz4.frame import decompress
import collections
import numpy as np

'''
分析训练集合和验证集合的 全部vocab，以及大于等于2的vocab 的重合程度，来分析MLM为啥效果不好。
分析训练集合和下游任务，IEMOCAP 和 MSP 等vocab的重合程度，来分析做好下游任务，需要多少数据，以及是否采用下游任务数据中加入预训练tasks。
有两个特殊符号需要注意一下：
    ## 为了标识一个subword是非首字符，会在该字符前面增加’##’， 字典中会存在很多 ## 的词
    @@ 字典中没有 @@ 开头的词，需要去掉 @@ 在vocab中找，一般后面是数字或者标点符号。

'''
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def get_word2num(txtdb_dir):
    txt2img_path = os.path.join(txtdb_dir, 'txt2img.json')
    txt2img =  json.load(open(txt2img_path))
    key_ids = txt2img.keys()
    print(len(key_ids))
    env = lmdb.open(txtdb1_dir)
    txn = env.begin(buffers=True)
    word2num = {}
    for key in key_ids:
        item = msgpack.loads(decompress(txn.get(key.encode('utf-8'))), raw=False)
        inputs = item.get('input_ids')
        tokens = item.get('toked_caption')
        assert len(inputs) == len(tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            if token.startswith('@@'):
                token = token.replace('@@', '')
            if word2num.get(token) is None:
                word2num[token] = 1
            else:
                word2num[token] += 1
    return word2num

txtdb1_dir = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.1_all_trn.db'
txtdb2_dir = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.1_all_2000.db'

print('[Debug] Read the info of training db')
trn_word2num = get_word2num(txtdb1_dir)
print('[Debug] Read the info of validation db')
val_word2num = get_word2num(txtdb2_dir)

print('[Debug] trn and val vocab diff')
trn_vocab = set(trn_word2num.keys())
val_vocab = set(val_word2num.keys())
overlap_vocab = val_vocab.intersection(trn_vocab)
print('trn {} val {} and overlap {}'.format(len(trn_vocab), len(val_vocab), len(overlap_vocab)))

more1_trn_vocab = [w for w in trn_word2num.keys() if trn_word2num[w] > 1]
more1_val_vocab = [w for w in val_word2num.keys() if val_word2num[w] > 1]
trn_vocab = set(more1_trn_vocab)
val_vocab = set(more1_val_vocab)
overlap_vocab = val_vocab.intersection(trn_vocab)
print('more1 trn {} val {} and overlap {}'.format(len(trn_vocab), len(val_vocab), len(overlap_vocab)))

vocab_path = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
word2index = load_vocab(vocab_path)
index2word = {value:key for key,value in word2index.items()}

print("-----mlm task words")
val_predict_indexs = np.load('./mlm_pred_words/val2000_predict_words.npy')
val_label_indexs = np.load('./mlm_pred_words/val2000_lable_words.npy')
correct_words = {}
error_words = {}
correct_words_num = 0
for i in range(len(val_predict_indexs)):
    word = index2word[val_predict_indexs[i]]
    if val_predict_indexs[i] == val_label_indexs[i]:
        if correct_words.get(word) is None:
            correct_words[word] = 1
        else:
            correct_words[word] += 1
        correct_words_num += 1
    else:
        if error_words.get(word) is None:
            error_words[word] = 1
        else:
            error_words[word] += 1
print('Val total {} words and {} correct'.format(len(val_predict_indexs), correct_words_num))
sort_correct_words = sorted(correct_words.items(), key=lambda d:d[1], reverse = True)
print(sort_correct_words)
print(len(sort_correct_words), correct_words_num/len(sort_correct_words)) # 253, avg=6.32
print("Error words info")
sort_error_words = sorted(error_words.items(), key=lambda d:d[1], reverse = True)
print(sort_error_words)
print(len(sort_error_words), (len(val_predict_indexs)-correct_words_num)/len(sort_error_words)) # 610, avg=2.96

print("------ analyse the words in training set")
print("For correct words in trn set")
trn_correct_words = {}
num_freq_correct = 0
for word in correct_words.keys():
    if trn_word2num.get(word) is None:
        print("word {} is not in trn set".format(word))
        continue
    trn_correct_words[word] = trn_word2num[word]
    num_freq_correct += trn_word2num[word]
print(trn_correct_words)
print(num_freq_correct/len(trn_correct_words))
print("For error words in trn set")
trn_error_words = {}
num_freq_error = 0
for word in error_words.keys():
    if trn_word2num.get(word) is None:
        print("word {} is not in trn set".format(word))
        continue
    trn_error_words[word] = trn_word2num[word]
    num_freq_error += trn_word2num[word]
print(trn_error_words)
print(num_freq_error/len(trn_error_words))