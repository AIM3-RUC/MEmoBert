'''
直接从txt_db数据中读取 input_ids 送入
'''
from preprocess.FileOps import read_csv
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)
from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
import json
import os
import numpy as np
import torch
import lmdb
import msgpack
from lz4.frame import decompress
import h5py
from tqdm import tqdm
from code.uniter.utils.save import ModelSaver
from code.bert_pretrain.run_cls import compute_metrics


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes, embd_method):
        super().__init__(config)
        self.num_labels = num_classes
        self.embd_method = embd_method
        if self.embd_method not in ['cls', 'mean', 'max']:
            raise NotImplementedError('Only [cls, mean, max] embd_method is supported, \
                but got', config.embd_method)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_layer = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.last_hidden_state
        cls_token = outputs.pooler_output
        hidden_states = outputs.hidden_states
        # using different embed method
        if self.embd_method == 'cls':
            cls_reps = cls_token
        elif self.embd_method == 'mean':
            cls_reps = torch.mean(last_hidden, dim=1)
        elif self.embd_method == 'max':
            cls_reps = torch.max(last_hidden, dim=1)[0]

        cls_reps = self.dropout(cls_reps)
        logits = self.cls_layer(cls_reps)
        return logits, hidden_states

def read_txt_db(txt_db_dir):
    env = lmdb.open(txt_db_dir)
    txn = env.begin(buffers=True)
    return txn

def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ids = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    # cls 101 sep 102
    device = torch.device('cuda:0')
    # model_dir = '/data7/emobert/exp/text_emo_model/bert_movie_model/' # num_classes=5
    # num_classes, model_postfix = 5, 'bert_movie'
    # model_dir = '/data7/emobert/exp/text_emo_model/all_5corpus_emo4_bert_base_lr2e-5_bs32/ckpt/' # num_classes=4
    # num_classes, model_postfix=4, 'all_5corpus_emo4'
    model_dir = '/data7/emobert/exp/text_emo_model/all_5corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1/' # num_classes=5
    num_classes, model_postfix =5, 'all_5corpus_emo5'
    
    model_saver = ModelSaver(model_dir)
    test_filepath = '/data7/emobert/'
    save_path = f'/data7/emobert/txt_pseudo_label/bert_movie_model-all_3corpus_val_txt_pseudo_label.h5'
    if 'all_5corpus' in  model_dir:
        config = AutoConfig.from_pretrained(model_dir, num_labels=num_classes)
        #print('config', config)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)   
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config)
    else:
        model = BertClassifier.from_pretrained(model_dir, num_classes=num_classes, embd_method='max')
    model_saver.save(model, 1)
    model.to(device)
    model.eval()

    all_target_emos = []
    all_pred_emos = []
    all_instances = read_csv(test_filepath, delimiter=',', skip_rows=1)
    save_h5f = h5py.File(save_path, 'w')
    index = 0
    for instance in tqdm(all_instances):
        sentence = instance[1]
        all_target_emos.append(int(instance[0]))
        cls_token = torch.tensor([[101]]).long()
        sep_token = torch.tensor([[102]]).long()
        input_ids = bert_tokenize(tokenizer, sentence)
        input_ids = torch.tensor([input_ids]).long()
        # need to add the special token, cls and sep
        input_ids = torch.cat([cls_token, input_ids, sep_token], dim=-1).to(device)
        masks = torch.ones(input_ids.size()).to(device)
        with torch.no_grad():
            if 'all_5corpus' in model_dir:
                # {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'sad', 4:'anger'}
                outputs = model.forward(input_ids)
                logits = outputs.logits
                pred = nn.functional.softmax(logits, dim=-1)
            else:
                # {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4}
                logits, _ = model.forward(input_ids, masks)
                pred = nn.functional.softmax(logits, dim=-1)
        all_pred_emos.append(torch.argmax(pred).cpu().numpy())
        # print(f'[Debug] sentence {sentence}')
        # print(f'[Debug] target {int(instance[0])}')
        # print(f'[Debug] pred {torch.argmax(pred).cpu().numpy()}')
        group = save_h5f.create_group(str(index))
        group['logits'] = logits.cpu().numpy()
        group['pred'] = pred.cpu().numpy()
        index += 1
    save_h5f.close()
    print(len(all_pred_emos), len(all_target_emos))
    assert len(all_pred_emos) == len(all_target_emos)
    results = compute_metrics(all_pred_emos, all_target_emos)
    print(results)

# export PYTHONPATH=/data7/MEmoBert
# CUDA_VISIBLE_DEVICES=1 python get_text_label.py