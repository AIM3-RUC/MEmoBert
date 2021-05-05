import re
import nltk
import os
import liwc
from collections import Counter
from preprocess.tools.bert_tokenization import load_vocab, WordpieceTokenizer
from preprocess.FileOps import read_pkl

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
        print(utt_tokens)
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
    
    def get_emo_words_by_tokens(self, utt_tokens):
        '''
        return:
            emo_words: all emotional words with category 'affect' 
            word2affect: word to emotion category 
            Jinming: word[token] = [anx, negemo]
        '''
        word2affect = {}
        emo_words = []
        for ind in range(len(utt_tokens)):
            token = utt_tokens[ind]
            categories = list(self.parse(utt_tokens[ind]))
            if 'affect' in categories:
                emo_words.append(token)
                for c in categories:
                    if c in self.emo_category_list:
                        if word2affect.get(token) is None:
                            word2affect[token] = [c]
                        else:
                            word2affect[token] += [c]
                # 如果不属于emo_category_list中的一类，那么就
                if word2affect.get(token) is None:
                    word2affect[token] = ['affect']
                word2affect[token] = list(set(word2affect[token]))
        return emo_words, word2affect

class EmoSentiWordLexicon():
    def __init__(self, word2score_path, bert_vocab_filepath):
        self.word2score = read_pkl(word2score_path)
        bert_vocab = load_vocab(bert_vocab_filepath)
        self.bert_tokenizer = WordpieceTokenizer(bert_vocab)
        # init sentiwordnet lookup/scoring tools
        self.impt = set(['NNS', 'NN', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
                    'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
                    'VBP', 'VBZ', 'unknown'])
        self.non_base = set(['VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NNS', 'NNPS'])
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.wnl = nltk.WordNetLemmatizer()

    def pos_short(self, pos):
        """Convert NLTK POS tags to SWN's POS tags."""
        if pos in set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
            return 'v'
        elif pos in set(['JJ', 'JJR', 'JJS']):
            return 'a'
        elif pos in set(['RB', 'RBR', 'RBS']):
            return 'r'
        elif pos in set(['NNS', 'NN', 'NNP', 'NNPS']):
            return 'n'
        else:
            return 'a'
    
    def score_word(self, word, pos):
        """Get sentiment score of word."""
        if self.word2score.get(word+'_'+pos) is None:
            # print(f'\t {word}_{pos} is not in word2score')
            return 0
        else:
            pos_s, neg_s = self.word2score.get(word+'_'+pos)
            # print(f'\t {word} {pos} pos-score {pos_s} neg_score {neg_s}' )
            return (pos_s - neg_s)

    def trans_categoty(self, score):
        # emo_category_list = {0:'noemoword', 1:'posemo', 2:'negemo'}
        if score > 0:
            category = 1
        elif score < 0:
            category = 2
        else:
            category = 0
        return category

    def score(self, utterance):
        """Sentiment score a sentence.
        utterance: utt-string or tokens-list
        https://github.com/anelachan/sentimentanalysis/blob/master/sentiment.py
        return:
        {word1:category, word2:category}
        """
        if isinstance(utterance, list):
            tokens = utterance
        else:
            tokens = list(self.bert_tokenizer.tokenize(utterance))
        word2emocate = {}
        # [('because', 'IN'), ('you', 'PRP'), ("##re", 'VBP'), ...]
        tagged = nltk.pos_tag(tokens)
        for el in tagged:
            ori_word, pos = el
            reobj = re.match('(\w+)', ori_word)
            if reobj is None:
                # print('reobj None: {} score =0 '.format(word))
                score = 0
            else:
                word = reobj.group(0).lower()
                if (pos in self.impt) and (word not in self.stopwords):
                    if pos in self.non_base:
                        word = self.wnl.lemmatize(word, self.pos_short(pos))
                    score = self.score_word(word, self.pos_short(pos))
                    # print(f'last {word} score {score}')
                else:
                    score = 0
                    # print('last {} score =0 '.format(word))
            category = self.trans_categoty(score)
            word2emocate[ori_word] = category
        return word2emocate

if __name__ == "__main__":
    # lexicon_dir = '/data2/zjm/tools/EmoLexicons'
    # lexicon_name = 'LIWC2015Dictionary.dic'
    # utterance = "Because you're going to get us all fucking pinched and embarrassing. What are you, so stupid?".lower()
    # bert_vocab_filepath = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
    # emol = EmoLexicon(lexicon_dir, lexicon_name, is_bert_token=True, bert_vocab_filepath=bert_vocab_filepath)
    # emo_words, word2affect = emol.get_emo_words(utterance)
    # print(word2affect)
    bert_vocab_filepath = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
    word2score_path = '/data2/zjm/tools/EmoLexicons/sentiword2score.pkl'
    utterance = "Because you're going to get us all fucking pinched and embarrassing. What are you, so stupid?".lower()
    emos = EmoSentiWordLexicon(word2score_path, bert_vocab_filepath)
    word2emo = emos.score(utterance)
    print(word2emo)