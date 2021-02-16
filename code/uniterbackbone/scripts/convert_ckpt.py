import sys
from collections import OrderedDict
import os
import torch

# step1 将 tf 转化为 torch
script_path = '/root/anaconda2/envs/vlbert/lib/python3.6/site-packages/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py'
tf_ckpt = '/data2/zjm/tools/LMs/bert_base_en/bert_model.ckpt'
config_path = '/data2/zjm/tools/LMs/bert_base_en/bert_config.json'
output_path = '/data2/zjm/tools/LMs/bert_base_en/bert_model_torch.pt'
if not os.path.exists(output_path):
    os.system('python {} --tf_checkpoint_path {} --bert_config_file {} --pytorch_dump_path {}'.format(script_path, \
                    tf_ckpt, config_path, output_path))

# step2 将 torch 中的key转化为 uniter
bert_ckpt = output_path
output_ckpt = '/data7/emobert/resources/pretrained/uniter-base-uncased-init.pt'
bert = torch.load(bert_ckpt)
uniter = OrderedDict()
for k, v in bert.items():
    uniter[k.replace('bert', 'uniter')] = v
torch.save(uniter, output_ckpt)