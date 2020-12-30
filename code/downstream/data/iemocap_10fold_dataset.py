import torch
import numpy as np
from os.path import join
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class Iemocap10foldDataset(data.Dataset):
    def __init__(self, opt, ft_dir, target_dir, setname='trn'):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__()
        self.exits_modality = {}
        if 'A' in opt.modality:
            acoustic_data = np.load(join(ft_dir, setname, "audio_ft.npy"))
            self.exits_modality['acoustic'] = acoustic_data

        if 'L' in opt.modality:
            lexical_data = np.load(join(ft_dir, setname, "txt_ft.npy"))
            self.exits_modality['lexical'] = lexical_data

        if 'V' in opt.modality:
            visual_data = np.load(join(ft_dir, setname, "face_ft.npy"))
            self.exits_modality['visual'] = visual_data

        self.label = np.load(join(target_dir, setname, "label.npy"))
        self.manual_collate_fn = True

    def __getitem__(self, index):
        '''
        if modalities =3, then example = {
                        'acoustic': acoustic, 
                        'lexical': lexical,
                        'visual': visual,
                        'label': label}
        '''
        example = {}
        for modal in self.exits_modality.keys():
            example[modal] = torch.from_numpy(self.exits_modality[modal][index])
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
        
        if 'lexical' in self.exits_modality.keys():
            L = [sample['lexical'] for sample in batches]
            L = pad_sequence(L, batch_first=True, padding_value=0)
            ret['lexical'] = L
        
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
    a = Iemocap10foldDataset(opt, ft_dir, ft_dir)
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
    