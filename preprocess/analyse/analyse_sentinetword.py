import os
import pickle as pkl
from nltk.corpus.reader.sentiwordnet import SentiWordNetCorpusReader

'''
https://github.com/nltk/nltk/blob/0a994230969d4487a658a315af1164fef849da70/nltk/corpus/reader/sentiwordnet.py#L44
step1: 分析包含的情感词的个数? 其中 词_pos1 词_pos2 作为两个词
    total 117659 neg 11693 pos 10219 neu 71022
step2: 相同的词有多少个？
    唯一的词, 62003, 正向的唯一的词 ，负向的唯一的词 
    pos uniwords 9242
    neg uniwords 10335
    neu uniwords 47149
'''

def read_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data

def write_pkl(filepath, data):
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)
    print("write {}".format(filepath))

def get_emo2score():
    swn = SentiWordNetCorpusReader(root=dic_root, fileids=[filename])
    all_keys = swn._db.keys() # 117659, key = POS_id
    emotion_word2score = {}
    pos_words_count = 0
    neg_words_count = 0
    neu_words_count = 0
    all_words = len(list(all_keys))
    for key in all_keys:
        syntem = swn.senti_synset(*key)
        name = syntem.synset.name() # able.a.01
        key = name.split('.')[0] + '_' +  name.split('.')[1]
        if emotion_word2score.get(key) is not None:
            continue
        pos_s = syntem.pos_score()
        neg_s = syntem.neg_score()
        if (pos_s - neg_s) > 0:
            pos_words_count += 1
        elif (pos_s - neg_s) < 0:
            neg_words_count += 1
        else:
            neu_words_count += 1
        emotion_word2score[key] = [pos_s, neg_s]
    print('total {} neg {} pos {} neu {}'.format(all_words, neg_words_count, pos_words_count, neu_words_count))
    write_pkl(save_path, emotion_word2score)
    print('emotion_word2score {}'.format(len(emotion_word2score)))

dic_root = '/data2/zjm/tools/EmoLexicons/'
save_path = '/data2/zjm/tools/EmoLexicons/sentiword2score.pkl'
filename = 'SentiWordNet_3.0.0.txt'
data = read_pkl(save_path)

uni_words = {}
uni_pos_words = {}
uni_neg_words = {}
uni_neu_words = {}
for key in data.keys():
    word = key.split('_')[0]
    uni_words[word] = 1
    pos_s, neg_s = data[key]
    if (pos_s - neg_s) > 0:
        uni_pos_words[word] = 1
    elif (pos_s - neg_s) < 0:
        uni_neg_words[word] = 1
    else:
        uni_neu_words[word] = 1
print('uniwords {}'.format(len(uni_words)))
print('pos uniwords {}'.format(len(uni_pos_words)))
print('neg uniwords {}'.format(len(uni_neg_words)))
print('neu uniwords {}'.format(len(uni_neu_words)))