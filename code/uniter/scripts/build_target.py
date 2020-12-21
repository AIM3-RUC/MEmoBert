import os
import json
import collections
import numpy as np

'''
构建finetune阶段的 target 信息
{'0':1, '1':2, '2':4}
'''

txt_db_path = '/data7/emobert/txt_db/movies_v1_th0.0_trn_2000.db/id2len.json'
target_path = '/data7/emobert/target/movies_v1/train_id2target.json'

id2target = collections.OrderedDict()
txt_db_id2len = json.load(open(txt_db_path))
for key in txt_db_id2len.keys():
    id2target[key] = 1
print(len(txt_db_id2len), len(id2target))
json.dump(id2target, open(target_path, 'w'))