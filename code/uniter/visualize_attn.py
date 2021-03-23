from collections import defaultdict
from os.path import join
from time import time
import math
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from horovod import torch as hvd
import os
from os.path import join, exists
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

import sys
sys.path.insert(0, '/data7/MEmoBert/')
from  code.uniter.data import (PrefetchLoader, TxtTokLmdb, ImageLmdbGroup, EmoCLsDataset,
                                emocls_collate)
from code.uniter.model.emocls import UniterForEmoRecognition, evaluation
from code.denseface.hook_demo import MultiLayerFeatureExtractor
from code.uniter.utils.misc import set_random_seed

import seaborn as sns
import matplotlib.pyplot as plt

emo_list = {0: 'neutral', 1:'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}

def build_dataloader(dataset, collate_fn, is_train=False):
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=is_train, drop_last=is_train,
                            num_workers=1,
                            pin_memory=True, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader

def read_pkl(filepath):
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
        return data

def write_pkl(filepath, data):
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)
    print("write {}".format(filepath))

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return lines

'''
可视化的是什么？ 可视化最后一层 CLS+文本+SEP+视觉 的权重分布。
1. 分析视觉信息对于最终的判决的作用。
2. 分析文本和视觉的对应关系。
'''

IMG_DIM = 342
layer_index = 11
model_config = "config/uniter-base-emoword_nomultitask.json"
checkpoint_dir = '/data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_4tasks-lr2e5_bs32_th0.5_train3000/drop0.1_frozen0_emocls_none'
checkpoint = checkpoint_dir + "/1/ckpt/model_step_600.pt"
vocab_path = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'

hvd.init()
n_gpu = hvd.size()
device = torch.device("cuda", hvd.local_rank())
torch.cuda.set_device(hvd.local_rank())
rank = hvd.rank()
#### loading model
set_random_seed(42)
checkpoint = torch.load(checkpoint)
model = UniterForEmoRecognition.from_pretrained(model_config, state_dict=checkpoint, \
                        img_dim=IMG_DIM, cls_num=7, frozen_en_layers=0, \
                        cls_dropout=0.1, cls_type='emocls')
model.to(device)
hook = MultiLayerFeatureExtractor(model, [f'uniter.encoder.layer.{layer_index}.attention.self.query', 
                                            f'uniter.encoder.layer.{layer_index}.attention.self.key',
                                            f'uniter.encoder.layer.{layer_index}.attention.self.value'])

all_img_dbs = ImageLmdbGroup(conf_th=0.1, max_bb=36, min_bb=10, num_bb=36, compress=False)
val_txt_db_path = '/data7/emobert/exp/evaluation/MELD/txt_db/1/val_emowords_emotype.db'
val_img_db_path = '/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/img_db/fc/'
val_img_db = all_img_dbs[val_img_db_path]
val_txt_db = TxtTokLmdb(val_txt_db_path, -1)
val_dataset = EmoCLsDataset(val_txt_db, val_img_db)
val_dataloader = build_dataloader(val_dataset, emocls_collate, False)

@torch.no_grad()
def evaluation(model, loader, visualization_info_path):
    visualization_info = {}
    total_pred = []
    total_target = []
    model.eval()
    eval_loss = 0
    # print('emo list {}'.format())
    for i, batch in enumerate(loader):
        print(f'\t \t [Info] For {i} sampe')
        img_frame_names = batch['img_frame_names']
        img_frame_name = img_frame_names[0]
        out = model(batch, compute_loss=False)
        loss = model.criterion(out, batch['targets'])
        eval_loss += loss.item()
        # the predicton reuslts
        preds = model.pred.argmax(dim=1).detach().cpu().numpy()
        targets = batch['targets'].detach().cpu().numpy()
        # for attention score
        query, key, value = hook.extract()
        # print(query.shape, key.shape, value.shape)
        attention_scores = torch.matmul(query, key.transpose(-1,-2))
        # attention_scores = np.matmul(query, np.transpose(key, (0, 2, 1)))
        attention_scores = attention_scores / math.sqrt(12)
        attention_scores = attention_scores.detach().cpu().numpy()
        # print('attention_scores {}'.format(attention_scores.shape))
        # for attention mask
        attention_mask = batch['attn_masks']
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).detach().cpu().numpy()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print('attention_mask {}'.format(extended_attention_mask.shape))
        attention_scores = attention_scores + extended_attention_mask
        attention_scores = attention_scores.squeeze().squeeze()
        # print('Finally attention_scores {}'.format(attention_scores.shape))
        # for using softmax, we can get prob
        attention_scores = torch.from_numpy(attention_scores)
        attention_probs = F.softmax(attention_scores,  dim=-1)
        # print(attention_probs.shape)
        visualization_info[img_frame_name] = {
            'input_ids': batch['input_ids'].detach().cpu().numpy(),
            'img_frame': img_frame_name,
            'pred': preds[0],
            'target': targets[0],
            'attention_probs': attention_probs,
        }
        total_pred.append(preds)
        total_target.append(targets)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_target)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    wf1 = f1_score(total_label, total_pred, average='weighted')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    print({'WA': acc, 'WF1': wf1, 'UA': uar,  'F1': f1})
    print(cm)
    write_pkl(visualization_info_path, visualization_info)


def visual_one_case(visualization_info_path, vocab_id2word, 
                save_dir, image_name='val_dia0_utt0', image_ind=0):
    # use image_name or image_ind
    visualization_info = read_pkl(visualization_info_path)
    image_names = list(visualization_info.keys())
    if image_name is None:
        image_name = image_names[image_ind]
        print('image_name is None and index {} is {}'.format(image_ind, image_name))
    
    save_path = join(save_dir, f'layer{layer_index}-' + image_name + '.png')
    data = visualization_info[image_name]
    atten_map = data['attention_probs']
    # get x and y tokens and frameindx
    dim_values = []
    input_ids = data['input_ids'][0]
    for i in range(len(atten_map)):
        if i < len(input_ids):
            dim_values.append(vocab_id2word[input_ids[i]])
        else:
            dim_values.append(i - len(input_ids))
    print(dim_values)
    plt.figure(figsize=(len(atten_map),len(atten_map)))
    axh1 = sns.heatmap(atten_map)#Seaborn对Pandas的对接比较好，可以直接处理DataFrame
    print('Target:{} Pred:{}'.format(emo_list[data['target']], emo_list[data['pred']]))
    axh1.set_title('Target:{} Pred:{}'.format(emo_list[data['target']], emo_list[data['pred']]))
    axh1.set_xticklabels(dim_values, rotation=90, fontsize=14)
    axh1.set_yticklabels(dim_values, rotation=0, fontsize=14)
    axh1.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    axh1.set_xlabel('txt_img')
    axh1.set_ylabel('txt_img')
    plt.savefig(save_path)


visualization_info_path = checkpoint_dir + f'/visualize_attmap_layer{layer_index}.pkl'

if False:
    evaluation(model, val_dataloader, visualization_info_path)

save_dir = checkpoint_dir + '/vis_pics'
if not exists(save_dir):
    os.mkdir(save_dir)

lines = read_file(vocab_path)
vocab_id2word = {i:line.strip() for i, line in enumerate(lines)}
print('vocab size {}'.format(len(vocab_id2word)))
for image_ind in range(100):
    visual_one_case(visualization_info_path, vocab_id2word, save_dir, image_name=None, image_ind=image_ind)

# 热度图可视化: https://blog.csdn.net/weixin_39541558/article/details/79813936