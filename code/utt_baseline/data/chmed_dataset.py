import torch
import numpy as np
from os.path import join
from torch._C import dtype
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class ChmedDataset(data.Dataset):
    def __init__(self, opt, ft_dir, target_dir, setname='train'):
        ''' MELD dataset reader
            no cross-validation
            set_name in ['train', 'val', 'test']
        '''
        super().__init__()
        self.opt = opt
        self.exits_modality = {}
        if 'A' in opt.modality:
            acoustic_data = np.load(join(ft_dir,  setname, "speech_{}_ft.npy".format(opt.a_ft_type)), allow_pickle=True)
            self.exits_modality['acoustic'] = acoustic_data

        if 'L' in opt.modality:
            # print(join(ft_dir,  setname, "text_{}_ft.npy".format(opt.l_ft_type)))
            text_data = np.load(join(ft_dir,  setname, "text_{}_ft.npy".format(opt.l_ft_type)), allow_pickle=True)
            self.exits_modality['text'] = text_data

        if 'V3d' in opt.modality:
            visual_data = np.load(join(ft_dir,  setname, "visual_{}_ft.npy".format(opt.v_ft_type)), allow_pickle=True)
            self.exits_modality['visual3d'] = visual_data
        elif 'V' in opt.modality:
            visual_data = np.load(join(ft_dir,  setname, "visual_{}_ft.npy".format(opt.v_ft_type)), allow_pickle=True)
            self.exits_modality['visual'] = visual_data

        self.label = np.load(join(target_dir,  setname, "label.npy"), allow_pickle=True)
        self.manual_collate_fn = True

    def __getitem__(self, index):
        '''
        # 跟 original dataset 是一样的，由于数据读取文件不同，所以这里也要重写。
        'max_text_tokens': 50,
        if modalities =3, then example = {
                        'acoustic': acoustic, 
                        'text': text,
                        'visual': visual,
                        'label': label}
        '''
        example = {}
        if 'acoustic' in self.exits_modality.keys():
            example['acoustic'] = torch.from_numpy(np.asarray(self.exits_modality['acoustic'][index], dtype=np.float32))
            if len(example['acoustic'].shape) > 1:
                if len(example['acoustic']) >= self.opt.max_acoustic_tokens:
                    example['acoustic'] = example['acoustic'][:self.opt.max_acoustic_tokens]
                else:
                    example['acoustic'] = torch.cat([example['acoustic'], \
                            torch.zeros([self.opt.max_acoustic_tokens-len(example['acoustic'])] + list(example['acoustic'].shape[1:]))], dim=0)

        if 'visual' in self.exits_modality.keys():
            example['visual'] = torch.from_numpy(np.asarray(self.exits_modality['visual'][index], dtype=np.float32))
            if len(example['visual'].shape) > 1:
                if len(example['visual']) >= self.opt.max_visual_tokens:
                    example['visual'] = example['visual'][:self.opt.max_visual_tokens]
                else:
                    example['visual'] = torch.cat([example['visual'], \
                            torch.zeros([self.opt.max_visual_tokens-len(example['visual'])] + list(example['visual'].shape[1:]))], dim=0)

        if 'visual3d' in self.exits_modality.keys():
            example['visual3d'] = torch.from_numpy(np.asarray(self.exits_modality['visual3d'][index], dtype=np.float32))
            if len(example['visual3d']) >= self.opt.max_visual_tokens:
                example['visual3d'] = example['visual3d'][:self.opt.max_visual_tokens]
            else:
                example['visual3d'] = torch.cat([example['visual3d'], \
                        torch.zeros([self.opt.max_visual_tokens-len(example['visual3d'])] + list(example['visual3d'].shape[1:]))], dim=0)
            # print(example['visual3d'].shape)
                        
        if 'text' in self.exits_modality.keys():
            example['text'] = torch.from_numpy(np.asarray(self.exits_modality['text'][index], dtype=np.float32))
            if len(example['text'].shape) > 1:
                if len(example['text']) >= self.opt.max_text_tokens:
                    example['text'] = example['text'][:self.opt.max_text_tokens]
                else:
                    example['text'] = torch.cat([example['text'], \
                            torch.zeros([self.opt.max_text_tokens-len(example['text'])] + list(example['text'].shape[1:]))], dim=0)
            
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
        
        if 'visual3d' in self.exits_modality.keys():
            V = [sample['visual3d'] for sample in batches]
            V = pad_sequence(V, batch_first=True, padding_value=0)
            ret['visual3d'] = V
        
        label = [sample['label'] for sample in batches]
        label = torch.tensor(label)
        ret["label"] = label
        return ret