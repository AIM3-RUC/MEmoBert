import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import numpy as np
import copy


class MSPimprovDataset(BaseDataset):
    def __init__(self, opt, set_name):
        ''' MSP-IMPROV dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)
        cvNo = opt.cvNo
        data_path = "/data6/lrc/MSP-IMPROV_feature/{}/cv_level/{}/"
        label_path = "/data6/lrc/MSP-IMPROV_feature/target/cv_level/{}/"

        self.acoustic_data = np.load(data_path.format('audio', cvNo) + f"{set_name}.npy")
        self.lexical_data = np.load(data_path.format('text', cvNo) + f"{set_name}.npy")
        self.visual_data = np.load(data_path.format('face', cvNo) + f"{set_name}.npy")

        self.label = np.load(label_path.format(cvNo) + f"{set_name}_label.npy")
        self.int2name = np.load(label_path.format(cvNo) + f"{set_name}_int2name.npy")
        print(f"MSP-IMPROV dataset {set_name} created with total length: {len(self)}")
    
    def __getitem__(self, index):
        acoustic = torch.from_numpy(self.acoustic_data[index])
        lexical = torch.from_numpy(self.lexical_data[index])
        visual = torch.from_numpy(self.visual_data[index])
        label = torch.tensor(self.label[index])
        index = torch.tensor(index)
        int2name = self.int2name[index]
        return {
            'acoustic': acoustic, 
            'lexical': lexical,
            'visual': visual,
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
    a = MSPimprovDataset(opt, 'val')
    data = next(iter(a))
    for k, v in data.items():
        if len(v.shape) == 0:
            print(k, v)
        else:
            print(k, v.shape)
