import os
import h5py
import numpy as np
import json
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

'''
建议使用新版的库 transformers抽特征
'''
class BertExtractorFromWWW(object):
    def __init__(self, cuda=False, cuda_num=None):
        # self.tokenizer = BertTokenizer.from_pretrained('/data2/lrc/bert_cache/pytorch')
        # self.model = BertModel.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output


def transform_utt_id(utt_id, set_name):
    dia_num, utt_num = utt_id.split('_')
    return f'{set_name}/dia{dia_num}_utt{utt_num}'

def get_trn_val_tst(target_root_dir, setname):
    int2name = np.load(os.path.join(target_root_dir, setname, 'int2name.npy'))
    int2label = np.load(os.path.join(target_root_dir, setname, 'label.npy'))
    int2name = [transform_utt_id(utt_id, setname) for utt_id in int2name[:, 0].tolist()]
    assert len(int2name) == len(int2label)
    return int2name, int2label

def make_T_feat():
    ref_root = '/data7/MEmoBert/evaluation/MELD/refs'
    save_root = '/data7/MEmoBert/evaluation/MELD/feature/bert_base'
    target_root = '/data7/MEmoBert/evaluation/MELD/target'
    bert_extractor = BertExtractorFromWWW(cuda=True, cuda_num=0)
   
    for set_name in ['train', 'val', 'test']:
        save_path = os.path.join(save_root, set_name + '.h5')
        h5f = h5py.File(save_path, 'w')
        json_data = json.load(open(os.path.join(ref_root, f'{set_name}.json')))
        int2name, _ = get_trn_val_tst(target_root, set_name)
        for utt_id in tqdm(int2name):
            text = json_data[utt_id]['txt'][0]
            feat, _ = bert_extractor.extract(text)
            feat = feat[0].cpu().numpy()
            h5f[utt_id] = feat
        h5f.close()

# make_T_feat()
if __name__ == '__main__':
    # bert_extractor = BertExtractor(gpu_id=0)
    # text = 'Ye is a feiwu.'
    # feat = bert_extractor.extract_feat(text)
    # print(feat.shape)
    make_T_feat()