import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class Iemocap10foldDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        acoustic_ft_type = opt.acoustic_ft_type
        lexical_ft_type = opt.lexical_ft_type
        visual_ft_type = opt.visual_ft_type
        data_path = "/data3/lrc/Iemocap_feature/cv_level/feature/{}/{}/"
        label_path = "/data3/lrc/Iemocap_feature/cv_level/target/{}/"

        self.acoustic_data = np.load(data_path.format(acoustic_ft_type, cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format(lexical_ft_type, cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format(visual_ft_type, cvNo) + f"{set_name}.npy")
        # mask for text feature
        self.l_mask = copy.deepcopy(self.lexical_data)
        self.l_mask[self.l_mask != 0] = 1
        self.v_mask = copy.deepcopy(self.lexical_data)
        self.v_mask[self.l_mask != 0] = 1
        self.l_length = self.mask2length(self.l_mask)
        self.v_length = self.mask2length(self.v_mask)
        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"IEMOCAP dataset {set_name} created with total length: {len(self)}")
    
    def mask2length(self, mask):
        ''' mask [total_num, seq_length, feature_size]
        '''
        _mask = np.mean(mask, axis=-1)        # [total_num, seq_length, ]
        length = np.sum(_mask, axis=-1)       # [total_num,] -> a number
        # length = np.expand_dims(length, 1)
        return length
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index][0].decode()
        # a_mask = torch.from_numpy(self.a_mask[index])
        v_mask = torch.from_numpy(self.v_mask[index])
        l_mask = torch.from_numpy(self.l_mask[index])
        l_length = torch.tensor(self.l_length[index])
        v_length = torch.tensor(self.v_length[index])
        return {
            'acoustic': acoustic, 
            'lexical': lexical,
            'visual': visual,
            'l_length': l_length,
            'v_length': v_length,
            'label': label,
            'index': index,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)

if __name__ == '__main__':
    class test:
        cvNo = 1
    
    opt = test()
    a = IemocapDataset(opt)
    print(next(iter(a)))