import sys
from collections import OrderedDict
import os
import torch

# step1 将 tf 转化为 torch
# script_path = '/root/anaconda2/envs/vlbert/lib/python3.6/site-packages/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py'
# model_type = 'bert_medium_en'
# tf_ckpt = '/data2/zjm/tools/LMs/{}/bert_model.ckpt'.format(model_type)
# config_path = '/data2/zjm/tools/LMs/{}/bert_config.json'.format(model_type)
# output_path = '/data2/zjm/tools/LMs/{}/bert_model_torch.pt'.format(model_type)
# if not os.path.exists(output_path):
#     os.system('python {} --tf_checkpoint_path {} --bert_config_file {} --pytorch_dump_path {}'.format(script_path, \
#                     tf_ckpt, config_path, output_path))

# step2 将 torch 中的key转化为 uniter
# bert_ckpt = '/data7/emobert/exp/text_emo_model/all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1/model_step_1.pt'
# output_ckpt = '/data7/MEmoBert/emobert/resources/pretrained/uniter_init_5corpus_emo5.pt'
bert_ckpt = '/data7/emobert/exp/text_emo_model/bert_movie_model/model_step_0.pt'
output_ckpt = '/data7/emobert/resources/pretrained/uniter_init_bertmovie.pt'
bert = torch.load(bert_ckpt)
uniter = OrderedDict()
for k, v in bert.items():
    uniter[k.replace('bert', 'uniter')] = v
torch.save(uniter, output_ckpt)