'''
直接从txt_db数据中读取 input_ids 送入
'''
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


if __name__ == '__main__':
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # ids = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    # print(ids)
    # input()
    # cls 101 sep 102
    corpus_name = 'voxceleb'
    version = 2
    setname = 'trn'  # only for IEMOCAP and MSP
    device = torch.device('cuda:0')
    # model_dir = '/data7/emobert/exp/text_emo_model/bert_movie_model/' # num_classes=5
    # num_classes, model_postfix = 5, 'bert_movie'
    # model_dir = '/data7/emobert/exp/text_emo_model/all_5corpus_emo4_bert_base_lr2e-5_bs32/ckpt/' # num_classes=4
    # num_classes, model_postfix=4, 'all_5corpus_emo4'
    model_dir = '/data7/emobert/exp/text_emo_model/all_3corpus_emo5_bert_base_lr2e-5_bs32_debug/ckpt/epoch-1/' # num_classes=5
    num_classes, model_postfix =5, 'corpus5_emo5'
    # model_dir = '/data7/emobert/exp/text_emo_model/all_5corpus_emo7_bert_base_lr2e-5_bs32/ckpt/' # num_classes=7
    # num_classes, model_postfix =7, 'all_5corpus_emo7'

    model_saver = ModelSaver(model_dir)
    if 'corpus' in  model_postfix:
        config = AutoConfig.from_pretrained(model_dir, num_labels=num_classes)
        #print('config', config)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)   
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config)
    else:
        model = BertClassifier.from_pretrained(model_dir, num_classes=num_classes, embd_method='max')
    model_saver.save(model, 0)
    model.to(device)
    model.eval()

    if corpus_name == 'movies':
        txt_db_dir = f'/data7/emobert/txt_db/movies_v{version}_th0.5_emowords_sentiword_all.db'
        save_path = f'/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_v{version}_{model_postfix}.h5'
    elif corpus_name == 'voxceleb':
        txt_db_dir = f'/data7/emobert/txt_db/voxceleb2_v{version}_th1.0_emowords_sentiword_all.db'
        save_path = f'/data7/emobert/txt_pseudo_label/voxceleb2_txt_pseudo_label_v{version}_{model_postfix}.h5'    
    elif corpus_name == 'opensub':
        txt_db_dir = f'/data7/emobert/txt_db/onlytext_opensub_p{version}_emo5_bert_data.db/'
        save_path = f'/data7/emobert/txt_pseudo_label/onlytext_opensub_p{version}_{model_postfix}.h5'    
    else:
        # for downstream tasks
        corpus_name_low = corpus_name.lower()
        txt_db_dir = f'/data7/emobert/exp/evaluation/{corpus_name}/txt_db/1/{setname}_emowords_sentiword.db/'
        save_path = f'/data7/emobert/txt_pseudo_label/{corpus_name_low}_txt_pseudo_label_{setname}_{model_postfix}.h5'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')
    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    print('total {} txts'.format(len(text2img)))

    save_h5f = h5py.File(save_path, 'w')
    for key in tqdm(text2img.keys()):
        item = msgpack.loads(decompress(txn.get(key.encode('utf-8'))), raw=False)
        cls_token = torch.tensor([[101]]).long()
        sep_token = torch.tensor([[102]]).long()
        input_ids = torch.tensor([item['input_ids']])
        # need to add the special token, cls and sep
        input_ids = torch.cat([cls_token, input_ids, sep_token], dim=-1).to(device)
        masks = torch.ones(input_ids.size()).to(device)
        with torch.no_grad():
            if 'corpus' in model_postfix:
                # {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'sad', 4:'anger'}
                outputs = model.forward(input_ids)
                logits = outputs.logits
                pred = nn.functional.softmax(logits, dim=-1)
            else:
                # {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4}
                logits, _ = model.forward(input_ids, masks)
                pred = nn.functional.softmax(logits, dim=-1)
        # print(item['toked_caption'])
        # print(pred)
        group = save_h5f.create_group(key)
        group['logits'] = logits.cpu().numpy()
        group['pred'] = pred.cpu().numpy()
    save_h5f.close()

# export PYTHONPATH=/data7/MEmoBert
# CUDA_VISIBLE_DEVICES=0 python get_text_weak_label.py