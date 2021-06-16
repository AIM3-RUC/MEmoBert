'''
统计一下几个数据集的数据长度，包括5个数据集，以及3个下游任务的数据集。
平均长度，小于等于3的，小于等于2的，大于等于10的。 由于文本太短的话

all_3corpus/emo5_bert_data/train.csv: 
avg_len 12.040020954823056 mid_len 10 m80_len 18 less2 4156 less3 7610 less10 37930 large3 64927
all_3corpus/emo5_bert_data/val.csv: 
avg_len 14.53352390852391 mid_len 12 m80_len 20 less2 152 less3 389 less10 4497 large3 11155
/data7/MEmoBert/emobert/exp/evaluation/IEMOCAP/bert_data/1/trn_val.csv
avg_len 12.191646682653877 mid_len 9 m80_len 19 less2 621 less3 920 less10 2868 large3 4084
/data7/MEmoBert/emobert/exp/evaluation/MSP/bert_data/1/trn_val.csv
avg_len 10.920388349514564 mid_len 9 m80_len 16 less2 319 less3 526 less10 2148 large3 3079

数据质量确实不怎么高，less2/less3的数据大约占10%的比例。要不要保留这些短的数据进行训练呢？
'''
from preprocess.FileOps import read_csv

def compute_stastic_info(lens):
    # 返回长度的中位数和80%分位点
    lens.sort()
    less10 = 0
    less3 = 0
    less2 = 0
    large3 = 0
    for l in lens:
        if l < 3:
            less2 += 1
        if l < 4:
            less3 += 1
        if l < 11:
            less10 += 1
        if l > 3:
            large3 += 1
    avg_len = sum(lens) / len(lens)
    mid_len = lens[int(len(lens)/2)]
    m80_len = lens[int(len(lens)*0.8)]
    return avg_len, mid_len, m80_len, less2, less3, less10, large3

bert_filepath = '/data7/MEmoBert/emobert/exp/evaluation/MSP/bert_data/1/trn_val.csv'
instances = read_csv(bert_filepath, delimiter=',')
lens = []
for instance in instances:
    sent = instance[1]
    lens.append(len(sent.split(' ')))
    
avg_len, mid_len, m80_len, less2, less3, less10, large3 = compute_stastic_info(lens)
print(f'avg_len {avg_len} mid_len {mid_len} m80_len {m80_len} less2 {less2} less3 {less3} less10 {less10} large3 {large3}')