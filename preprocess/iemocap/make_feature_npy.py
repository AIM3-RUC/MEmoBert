import os
import h5py
import numpy as np
import json
from tqdm import tqdm
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer

class BertExtractor(object):
    def __init__(self, gpu_id=None):
        self.device = torch.device(f'cuda:{gpu_id}') if gpu_id is not None else None
        self.model = BertModel.from_pretrained('bert-base-uncased')
        if self.device:
            self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    def extract_feat(self, text):
        ids = self.bert_tokenize(text)
        ids = torch.tensor(ids).unsqueeze(0)
        if self.device:
            ids = ids.to(self.device)
        with torch.no_grad():
            feat = self.model(ids)[0][0]
        return feat.squeeze().cpu().numpy()

    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids

def get_trn_val_tst(cv, target_root='target'):
    target_dir = os.path.join(target_root, str(cv))
    trn_int2name = np.load(os.path.join(target_dir, 'trn_int2name.npy'))
    val_int2name = np.load(os.path.join(target_dir, 'val_int2name.npy'))
    tst_int2name = np.load(os.path.join(target_dir, 'tst_int2name.npy'))
    trn_label = np.load(os.path.join(target_dir, 'trn_label.npy'))
    val_label = np.load(os.path.join(target_dir, 'val_label.npy'))
    tst_label = np.load(os.path.join(target_dir, 'tst_label.npy'))
    assert len(trn_int2name) == len(trn_label)
    assert len(val_int2name) == len(val_label)
    assert len(tst_int2name) == len(tst_label)
    return trn_int2name, val_int2name, tst_int2name, trn_label, val_label, tst_label

def make_T_feat():
    ref_root = '/data7/MEmoBert/evaluation/IEMOCAP/refs'
    save_root =  '/data7/MEmoBert/evaluation/IEMOCAP/feature/bert'
    bert_extractor = BertExtractor(gpu_id=0)
    for cv in range(1, 11):
        trn_int2name, val_int2name, tst_int2name, _, _, _ = get_trn_val_tst(cv, \
            '/data7/MEmoBert/evaluation/IEMOCAP/target')
        int2name_lookup = {
            'trn': trn_int2name,
            'val': val_int2name,
            'tst': tst_int2name
        }
        for part in ['trn', 'val', 'tst']:
            save_path = os.path.join(save_root, str(cv))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, f'{part}.h5')
            h5f = h5py.File(save_path, 'w')
            json_data = json.load(open(os.path.join(ref_root, str(cv), f'{part}_ref.json')))
            print(f"CV:{cv} Part:{part}")
            for utt_id in tqdm(int2name_lookup[part]):
                utt_id = utt_id[0].decode('utf8')
                text = json_data[utt_id]['txt'][0]
                feat = bert_extractor.extract_feat(text)
                h5f[utt_id] = feat
            h5f.close()

# make_T_feat()
if __name__ == '__main__':
    # bert_extractor = BertExtractor(gpu_id=0)
    # text = 'Ye is a feiwu.'
    # feat = bert_extractor.extract_feat(text)
    # print(feat.shape)
    make_T_feat()