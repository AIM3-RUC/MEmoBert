import csv
from preprocess.FileOps import read_file, read_csv, write_csv

# 将1000w的文本数据分成4份，每份250w, 然后作为单独文本的数据加入训练
# p1: 2271927
# p2: 2254491
# p3: 2252085
# p4: 1903143

text_filepath = '/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/OpenSubtitles.en-zh_cn.en_trn1000w.txt'
start = 2500000 * 3
end = 2500000 * 4

# emo column 统一填充为0
csv_filepath = '/data7/emobert/exp/mlm_pretrain/datasets/OpenSubtitlesV2018/opensub1000w_p4.csv'

all_instances = []
lines = read_file(text_filepath)
for line in lines[start:end]:
    if len(line.split(' ')) > 2:
        all_instances.append([0, line.strip('\n\t')])
print(csv_filepath)
print('all_instances {}'.format(len(all_instances)))
write_csv(csv_filepath, all_instances, delimiter=',')