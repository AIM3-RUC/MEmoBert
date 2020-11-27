import re
import nltk
import os
import liwc
from collections import Counter
from emobert.preprocess.tools.bert_tokenization import load_vocab, WordpieceTokenizer

'''
根据情感词典，获取句子中的情感词，三个不同的词典
分别是 LIWC, SentiWordNet, VAD 标注
其中先以 LIWC 为例: https://github.com/chbrown/liwc-python 
采用wordpiece进行分词的话，会导致词非常细粒度, 跟Bert保持一致.
'''

class EmoLexicon():
    def __init__(self, lexicon_dir, lexicon_name, is_bert_token=True, bert_vocab_filepath=None):
        self.lexicon_dir = lexicon_dir
        self.lexicon_name = lexicon_name
        if 'LIWC' in self.lexicon_name:
            self.parse, self.category_names = liwc.load_token_parser(os.path.join(self.lexicon_dir, self.lexicon_name))
            # print('there are {} catogories in LIWC'.format(len(self.category_names)))
            # print(self.category_names)
        else:
            print('The {} is not implemented'.format(lexicon_name))
            exit(0)
        self.emo_category_list = ['posemo', 'negemo', 'anx', 'anger', 'sad']
        
        self.is_bert_token = is_bert_token
        if self.is_bert_token:
            bert_vocab = load_vocab(bert_vocab_filepath)
            self.bert_tokenizer = WordpieceTokenizer(bert_vocab)

    def tokenize(self, utterance):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', utterance, re.UNICODE):
            yield match.group(0)        
    
    def get_emo_words(self, utterance):
        '''
        return:
            emo_words: all emotional words with category 'affect' 
            word2affect: word to emotion category
        '''
        word2affect = {}
        emo_words = []
        if self.is_bert_token:
            utt_tokens = list(self.bert_tokenizer.tokenize(utterance))
        else:
            utt_tokens = list(self.tokenize(utterance))
        # print(utt_tokens)
        for ind in range(len(utt_tokens)):
            token = utt_tokens[ind]
            categories = list(self.parse(utt_tokens[ind]))
            if 'affect' in categories:
                emo_words.append(token)
                word2affect[token] = 'affect'
                for c in categories:
                    if c in self.emo_category_list:
                        word2affect[token] = c
        return emo_words, word2affect

if __name__ == "__main__":
    lexicon_dir = '/data2/zjm/tools/EmoLexicons'
    lexicon_name = 'LIWC2015Dictionary.dic'
    utterance = "Because you're going to get us all fucking pinched. What are you, so stupid?".lower()
    bert_vocab_filepath = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
    emol = EmoLexicon(lexicon_dir, lexicon_name, is_bert_token=True, bert_vocab_filepath=bert_vocab_filepath)
    emo_words, word2affect = emol.get_emo_words(utterance)
    print(word2affect)