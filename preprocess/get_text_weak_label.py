'''
直接从txt_db数据中读取 input_ids 送入
'''
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

class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes, embd_method): #
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
    version = 3
    device = torch.device('cuda:0')
    model_dir = '/data10/lrc/movie_dataset/pretrained_model/bert_movie_model'
    model = BertClassifier.from_pretrained(model_dir, num_classes=5, embd_method='max')
    model.to(device)
    model.eval()
    txt_db_dir = f'/data7/MEmoBert/emobert/txt_db/movies_v{version}_th0.5_emowords_sentiword_all.db'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')
    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    print('total {} txts'.format(len(text2img)))
    save_path = f'/data7/MEmoBert/emobert/txt_pseudo_label/movie_txt_pseudo_label_v{version}.h5'
    save_h5f = h5py.File(save_path, 'w')

    for key in tqdm(text2img.keys()):
        item = msgpack.loads(decompress(txn.get(key.encode('utf-8'))), raw=False)
        # {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4}
        cls_token = torch.tensor([[101]]).long()
        sep_token = torch.tensor([[102]]).long()
        input_ids = torch.tensor([item['input_ids']])
        # need to add the special token, cls and sep
        input_ids = torch.cat([cls_token, input_ids, sep_token], dim=-1).to(device)
        masks = torch.ones(input_ids.size()).to(device)
        with torch.no_grad():
            logits, _ = model.forward(input_ids, masks)
            pred = nn.functional.softmax(logits, dim=-1)
        
        # print(item['toked_caption'])
        # print(logits)
        # print(pred)
        # input()
        group = save_h5f.create_group(key)
        group['logits'] = logits.cpu().numpy()
        group['pred'] = pred.cpu().numpy()
    
    save_h5f.close()
