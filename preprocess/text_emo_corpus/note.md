根据整理的文本情感数据，构建一个规模的可用的文本情感标注数据，总量大约在20w左右。

之前用的训练数据是，7分类统一命名  neutral + happy + surprise + sad + anger + fear + disgust
train: 21672 val: 2846 test: 6004 
总的数据规模在 3w.

/data7/emobert/text_emo_corpus

## 1. 测试一下 MELD MELD 还是有效果的
## 2. balance 一下类别数据
## 3. 预训练的时候的EmoCls单独做一个数据集合（只用质量高的数据做）neu 概率大于 80% 其他类别的概率大于 40%. 

## 整理下面5个情感数据集，
emo_list = {0: "neutral", 1: "happy", 2: "surprise", 3: "sad", 4:"anger", 5: "fear", 6: "disgust"}
emo_list = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'sad', 4:'anger'}
emo_list = {0: 'anger', 1: 'happy', 2: 'neutral', 3: 'sadness'}

总的数据规模以及情感分布情况
train 162515 val 13819
train emo neutral sents 109009 --to 2w
train emo surprise sents 5813
train emo fear sents 4021
train emo happy sents 26994 -- to 2w
train emo sadness sents 5944
train emo anger sents 7897
train emo disgust sents 2837
test emo surprise sents 535
test emo neutral sents 9239
test emo happy sents 2501
test emo anger sents 555
test emo fear sents 282
test emo disgust sents 194
test emo sadness sents 513

1. emo7_bert_data: set train 66515 set val 13820
    {'neutral': 0, 'happy': 1, 'surprise': 2, 'sadness': 3, 'anger': 4, 'fear': 5, 'disgust': 6}
    train emo neutral sents 20001
    train emo surprise sents 5813
    train emo fear sents 4021
    train emo happy sents 20001
    train emo sadness sents 5944
    train emo anger sents 7897
    train emo disgust sents 2837
    set train 66515
    train emo surprise sents 535
    train emo neutral sents 9239
    train emo happy sents 2501
    train emo anger sents 555
    train emo fear sents 282
    train emo disgust sents 194
    train emo sadness sents 513
    set val 13820
2. emo5_bert_data: set train 59657 set val 13344
    {'neutral': 0, 'happy': 1, 'surprise': 2, 'sadness': 3, 'anger': 4}
    train emo neutral sents 20001
    train emo surprise sents 5813
    train emo happy sents 20001
    train emo sadness sents 5944
    train emo anger sents 7897
    set train 59657
    train emo surprise sents 535
    train emo neutral sents 9239
    train emo happy sents 2501
    train emo anger sents 555
    train emo sadness sents 513
    set val 13344
2. emo4_bert_data: set train 178470 set val 9393
    {'anger': 0, 'happy': 1, 'neutral': 2, 'sadness': 3}
    train emo neutral sents 20001
    train emo happy sents 20001
    train emo sadness sents 5944
    train emo anger sents 7897
    set train 53844
    train emo neutral sents 9239
    train emo happy sents 2501
    train emo anger sents 555
    train emo sadness sents 513
    set val 12809
    
# 1. 整理 EmotionLine
原始数据地址：
/data3/zjm/dataset/EmotionLines/Friends/{friends_dev.json, friends_test.json, friends_train.json}
类别信息:
[neutral, joy, sadness, fear, anger, surprise, disgust]
joy -> happy

合并train+val+test三个数据集，总数据量: 10001
数据分布: 
train emo neutral sents 4462
train emo surprise sents 1062
train emo fear sents 199
train emo happy sents 1205
train emo sadness sents 397
train emo anger sents 536
train emo disgust sents 248
test emo surprise sents 219
test emo neutral sents 1107
test emo happy sents 245
test emo anger sents 146
test emo fear sents 29
test emo disgust sents 63
test emo sadness sents 83
all trainval instance 8109
all test instance 1892

# 2. 整理 XED
/data3/zjm/dataset/XED/{en-annotated.tsv, neu_en.txt}
{"anger":1, "anticipation":2, "disgust":3, "fear":4, "joy":5, "sadness":6, "surprise":7, "trust":8, "neutral":0}

多标签分类的情况，比如是 happy-surprise angry-surprise 那么不能同时保留两个标签。
因为 happy 和 angry 会混淆. 而在下游任务中我们是将 surprise 和 happy 合并为 happy 的.

总数据量: 21665
数据分布: 
17528 /data3/zjm/dataset/XED/en-annotated.tsv
9675 /data3/zjm/dataset/XED/neu_en.txt
emo anger sents 3669
emo happy sents 2825
emo disgust sents 1633
emo fear sents 1784
emo sadness sents 1702
emo surprise sents 1426
emo neutral sents 8626
all instance 21665

# 3. 整理 emorynlp
/data3/zjm/dataset/emorynlp

情感做一个映射, 可用的有5类别，anger fear joy sadness neutral:    
emomap = {"Peaceful": 'peaceful', "Mad":'anger', "Scared":'fear', "Joyful":'joy', "Sad": 'sadness', "Powerful":'powerful',  "Neutral":'neutral'}

总数据量: 8971
数据分布: 
train emo happy sents 2199
train emo neutral sents 2775
train emo anger sents 1130
train emo sadness sents 690
train emo fear sents 1302
test emo anger sents 102
test emo neutral sents 265
test emo happy sents 251
test emo fear sents 162
test emo sadness sents 95
all trainval instance 8096
all test instance 875

# 4. 整理 DailyDialog
/data3/zjm/dataset/dailydialog
{0:neutral , 1: anger, 2: disgust, 3: fear, 4: joy, 5: sadness, 6: surprise}

__eou__ 是句子分隔符
合并后的数据不对，需要用原始的不同Set中的数据

总数据量: 102753
数据分布 七类情感: 
train emo neutral sents 79066
train emo happy sents 11846
train emo surprise sents 1706
train emo fear sents 157
train emo disgust sents 306
train emo sadness sents 1048
train emo anger sents 904
test emo neutral sents 6303
test emo surprise sents 116
test emo fear sents 17
test emo happy sents 1017
test emo sadness sents 102
test emo anger sents 118
test emo disgust sents 47
all trainval instance 95033
all test instance 7720


# 5. 整理 GoEmotions
/data3/zjm/dataset/goemotions/processed_data

{
"anger": ["anger", "annoyance", "disapproval"],
"disgust": ["disgust"],
"fear": ["fear", "nervousness"],
"joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
"sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
"surprise": ["surprise", "realization", "confusion", "curiosity"]
}

总数据量: 29612 + 3332
数据分布 七类情感: 
train emo neutral sents 14080
train emo fear sents 579
train emo surprise sents 1619
train emo happy sents 8919
train emo anger sents 1658
train emo sadness sents 2107
train emo disgust sents 650
test emo sadness sents 233
test emo happy sents 988
test emo neutral sents 1564
test emo fear sents 74
test emo anger sents 189
test emo surprise sents 200
test emo disgust sents 84
all trainval instance 29612
all test instance 3332