import torch
import numpy as np
from os.path import join
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class MspPretrainedDataset(data.Dataset):
    def __init__(self, opt, ft_dir, target_dir, setname_str='trn'):
        ''' MSP dataset reader
        extracting features from pretrained emobert model 
            set_name in ['trn', 'val', 'tst']
        if trn,val, then combine the trn and val datasets 
        '''
        super().__init__()
        self.opt = opt
        setnames = setname_str.split(',')
        self.exits_modality = {}
        if 'A' in opt.modality:
            acoustic_data = []
            for setname in setnames:
                temp = np.load(join(ft_dir, str(opt.cvNo), setname, "speech_ft.npy"), allow_pickle=True)
                acoustic_data.extend(list(temp))
            self.exits_modality['acoustic'] = acoustic_data

        if 'L' in opt.modality:
            text_data = []
            for setname in setnames:
                temp = np.load(join(ft_dir, str(opt.cvNo), setname, "txt_ft.npy"), allow_pickle=True)
                text_data.extend(list(temp))
            self.exits_modality['text'] = text_data

        if 'V' in opt.modality:
            visual_data = []
            for setname in setnames:
                temp = np.load(join(ft_dir, str(opt.cvNo), setname, "img_ft.npy"), allow_pickle=True)
                visual_data.extend(list(temp))
            self.exits_modality['visual'] = visual_data

        self.label = []
        for setname in setnames:
            temp = np.load(join(target_dir, str(opt.cvNo), setname, "label.npy"), allow_pickle=True)
            self.label.extend(list(temp))

        self.manual_collate_fn = True

    def __getitem__(self, index):
        '''
        # 跟 original dataset 是一样的，由于数据读取文件不同，所以这里也要重写。
        'max_lexical_tokens': 50,
        if modalities =3, then example = {
                        'acoustic': acoustic, 
                        'lexical': lexical,
                        'visual': visual,
                        'label': label}
        '''
        example = {}
        if 'acoustic' in self.exits_modality.keys():
            example['acoustic'] = torch.from_numpy(self.exits_modality['acoustic'][index])
            if len(example['acoustic']) >= self.opt.max_acoustic_tokens:
                example['acoustic'] = example['acoustic'][:self.opt.max_acoustic_tokens]
            else:
                example['acoustic'] = torch.cat([example['acoustic'], \
                        torch.zeros([self.opt.max_acoustic_tokens-len(example['acoustic']), self.opt.a_input_size])], dim=0)
        if 'visual' in self.exits_modality.keys():
            try:
                example['visual'] = torch.from_numpy(self.exits_modality['visual'][index])
            except ValueError:
                example['visual'] = torch.zeros(1, self.opt.v_input_size)
            if len(example['visual']) >= self.opt.max_visual_tokens:
                example['visual'] = example['visual'][:self.opt.max_visual_tokens]
            else:
                example['visual'] = torch.cat([example['visual'], \
                        torch.zeros([self.opt.max_visual_tokens-len(example['visual']), self.opt.v_input_size])], dim=0)

        if 'text' in self.exits_modality.keys():
            example['text'] = torch.from_numpy(self.exits_modality['text'][index])
            if len(example['text']) >= self.opt.max_text_tokens:
                example['text'] = example['text'][:self.opt.max_text_tokens]
            else:
                example['text'] = torch.cat([example['text'], \
                        torch.zeros([self.opt.max_text_tokens-len(example['text']), self.opt.l_input_size])], dim=0)
        
        label = torch.tensor(self.label[index])
        example['label'] = label

        return example
    
    def __len__(self):
        return len(self.label)
    
    def collate_fn(self, batches):
        ret = {}
        if 'acoustic' in self.exits_modality.keys():
            A = [sample['acoustic'] for sample in batches]
            A = pad_sequence(A, batch_first=True, padding_value=0)
            ret['acoustic'] = A

        if 'text' in self.exits_modality.keys():
            L = [sample['text'] for sample in batches]
            L = pad_sequence(L, batch_first=True, padding_value=0)
            ret['text'] = L
        
        if 'visual' in self.exits_modality.keys():
            V = [sample['visual'] for sample in batches]
            V = pad_sequence(V, batch_first=True, padding_value=0)
            ret['visual'] = V
        
        label = [sample['label'] for sample in batches]
        label = torch.tensor(label)
        ret["label"] = label
        return ret

if __name__ == '__main__':
    class test:
        modality = 'VL'
    opt = test()
    ft_dir = '/data7/MEmoBert/emobert/exp/mmfts/iemocap/nomask_movies_v1_uniter_4tasks_nofinetune/1'
    a = IemocapPretrainedDataset(opt, ft_dir, ft_dir)
    # print(next(iter(a)))
    _iter = iter(a)
    data1 = next(_iter)
    data2 = next(_iter)
    data3 = next(_iter)
    
    for k, v in data1.items():
        print(k, v.shape)
    print()
    for k, v in data2.items():
        print(k, v.shape)
    print()
    for k, v in data3.items():
        print(k, v.shape)
    print()

    kk = a.collate_fn([data1, data2, data3])
    for k, v in kk.items():
        print(k, v.shape)
    