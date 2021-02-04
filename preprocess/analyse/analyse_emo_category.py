import os
import json
import liwc
from pytorch_pretrained_bert import BertTokenizer
import lmdb
import msgpack
from lz4.frame import decompress

'''
export PYTHONPATH=/data7/MEmoBert
分析情感词的情况
1. 统计情感词典中常见的情感类别对应的词的个数，并排序，找出现次数最高的几个情感类别。
2. 统计数据集中出现的情感词的分布，主要是集中的哪些类别？
'''

def analyse_emo_dic():
    lexicon_dir = '/data2/zjm/tools/EmoLexicons'
    lexicon_name = 'LIWC2015Dictionary.dic'
    ### basic useful
    parse, category_names = liwc.load_token_parser(os.path.join(lexicon_dir, lexicon_name))
    print(category_names)
    print(len(category_names))
    print(list(parse('happy'))) # ['adj', 'affect', 'posemo']
    lexicon, category_names = liwc.dic.read_dic(os.path.join(lexicon_dir, lexicon_name))
    print(len(lexicon))
    category2words = {}
    for key in lexicon.keys():
        emos = lexicon[key]
        for e in emos:
            if category2words.get(e) is None:
                category2words[e] = [key]
            else:
                category2words[e] += [key]
    print(len(category2words))
    category2num_words = {}
    for emo in category2words.keys():
        category2num_words[emo] = len(category2words[emo])
    sorted_emo2num_words = sorted(category2num_words.items(), key=lambda d:d[1], reverse = True)
    print(sorted_emo2num_words)

    affect_words = category2words['affect']
    posemo_words = category2words['posemo']
    negemo_words = category2words['negemo']
    print(len(posemo_words), len(negemo_words), type(posemo_words))
    combine_words  = []
    combine_words.extend(posemo_words)
    combine_words.extend(negemo_words)
    diff_words = set(affect_words).difference(combine_words)
    print(diff_words)
    for word in diff_words:
        print(word, ':\t', lexicon[word])
    # posaffect_words = list(set(affect_words).intersection(set(posemo_words)))
    # negaffect_words = list(set(affect_words).intersection(set(negemo_words)))
    # print(len(posaffect_words))
    # print(len(negaffect_words))
    # ang_words = category2words['anger']
    # anx_words = category2words['anx']
    # sad_words = category2words['sad']
    # angneg_words = list(set(negemo_words).intersection(set(ang_words)))
    # anxneg_words = list(set(negemo_words).intersection(set(anx_words)))
    # sadneg_words = list(set(negemo_words).intersection(set(sad_words)))
    # print(len(angneg_words), len(anxneg_words), len(sadneg_words))    

if __name__ == '__main__':    

    # analyse_emo_dic()
    txtdb_dir = '/data7/MEmoBert/emobert/txt_db/movies_v1_th0.0_emowords_trn.db'
    txt2img_path = os.path.join(txtdb_dir, 'txt2img.json')
    txt2img =  json.load(open(txt2img_path))
    key_ids = txt2img.keys()
    print(len(key_ids))
    env = lmdb.open(txtdb_dir)
    txn = env.begin(buffers=True)
    emotional_utt = 0
    emos2num = {}
    emos_words2num = {}
    for key in key_ids:
        item = msgpack.loads(decompress(txn.get(key.encode('utf-8'))), raw=False)
        if len(item.get('emo_labels')) > 0:
            inputs = item.get('input_ids')
            tokens = item.get('toked_caption')
            assert len(inputs) == len(tokens)
            for input_id in item.get('emo_input_ids'):
                index = inputs.index(input_id) 
                token = tokens[index]
                if emos_words2num.get(token) is None:
                    emos_words2num[token] = 1
                else:
                    emos_words2num[token] += 1
            emotional_utt += 1
            for emo in item.get('emo_labels'):
                if emos2num.get(emo) is None:
                    emos2num[emo] = 1
                else:
                    emos2num[emo] += 1
    print('emotional utts {}'.format(emotional_utt))
    sorted_emos2num = sorted(emos2num.items(), key=lambda d:d[1], reverse = True)
    print(sorted_emos2num)
    sorted_emos_words2num = sorted(emos_words2num.items(), key=lambda d:d[1], reverse = True)
    print(sorted_emos_words2num)
    print(len(sorted_emos_words2num))