import torch
import numpy as np
from os.path import join
import h5py
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

class IemocapOriginalDataset(data.Dataset):
    def __init__(self, opt, ft_dir, target_dir, setname='trn'):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
            ft_dir: /data7/MEmoBert/evaluation/IEMOCAP/feature
            target_dir: /data7/MEmoBert/evaluation/IEMOCAP/target
        '''
        super().__init__()
        self.opt = opt
        self.exits_modality = {}
        # add by zjm 15/01/2021: need to modify if add audio and text
        self.opt.v_ft_name = self.opt.pretained_ft_type
        A_feat_dir = join(ft_dir, self.opt.a_ft_name, str(opt.cvNo))
        V_feat_dir = join(ft_dir, self.opt.v_ft_name, str(opt.cvNo))
        L_feat_dir = join(ft_dir, self.opt.l_ft_name, str(opt.cvNo))

        if 'A' in opt.modality:
            acoustic_data = h5py.File(join(A_feat_dir, setname + '.h5'), 'r')
            self.exits_modality['acoustic'] = acoustic_data
            print('[Afeat-dir] {}'.format(A_feat_dir))

        if 'L' in opt.modality:
            lexical_data = h5py.File(join(L_feat_dir, setname + '.h5'), 'r')
            self.exits_modality['lexical'] = lexical_data
            print('[Lfeat-dir] {}'.format(L_feat_dir))

        if 'V' in opt.modality:
            visual_data = h5py.File(join(V_feat_dir, setname + '.h5'), 'r')
            self.exits_modality['visual'] = visual_data
            print('[Vfeat-dir] {}'.format(V_feat_dir))

        self.int2name = np.load(join(target_dir, str(opt.cvNo), f"{setname}_int2name.npy"))
        self.label = np.load(join(target_dir, str(opt.cvNo), f"{setname}_label.npy"))
        if len(self.label.shape) > 1:
            self.label = np.argmax(self.label, axis=1)            
        self.manual_collate_fn = False

    def __getitem__(self, index):
        '''
        read from h5py 
        if modalities =3, then example = {
                        'acoustic': acoustic, 
                        'lexical': lexical,
                        'visual': visual,
                        'label': label}
        '''
        example = {}
        try:
            # for iemocap that the int2name is binary type
            utt_id = self.int2name[index][0].decode('utf8')
        except:
            utt_id = self.int2name[index]
        if 'acoustic' in self.exits_modality.keys():
            example['acoustic'] = torch.from_numpy(self.exits_modality['acoustic'][utt_id]['feat'][()])
            if len(example['acoustic']) >= self.opt.max_acoustic_tokens:
                example['acoustic'] = example['acoustic'][:self.opt.max_acoustic_tokens]
            else:
                example['acoustic'] = torch.cat([example['acoustic'], \
                        torch.zeros([self.opt.max_acoustic_tokens-len(example['acoustic']), self.opt.a_input_size])], dim=0)

        if 'visual' in self.exits_modality.keys():
            try:
                example['visual'] = torch.from_numpy(self.exits_modality['visual'][utt_id]['feat'][()])
            except ValueError:
                example['visual'] = torch.zeros(1, self.opt.v_input_size)
            if len(example['visual']) >= self.opt.max_visual_tokens:
                example['visual'] = example['visual'][:self.opt.max_visual_tokens]
            else:
                example['visual'] = torch.cat([example['visual'], \
                        torch.zeros([self.opt.max_visual_tokens-len(example['visual']), self.opt.v_input_size])], dim=0)

        if 'lexical' in self.exits_modality.keys():
            example['lexical'] = torch.from_numpy(self.exits_modality['lexical'][utt_id][()])
            if len(list(example['lexical'].size())) == 1:
                example['lexical'] = example['lexical'].reshape([-1, self.opt.l_input_size])
            # print(list(example['lexical'].size()))
            if len(example['lexical']) >= self.opt.max_lexical_tokens:
                example['lexical'] = example['lexical'][:self.opt.max_lexical_tokens]
            else:
                example['lexical'] = torch.cat([example['lexical'], \
                        torch.zeros([self.opt.max_lexical_tokens-len(example['lexical']), self.opt.l_input_size])], dim=0)
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
        cvNo = 1 
    opt = test()
    ft_dir = "/data7/MEmoBert/evaluation/IEMOCAP/feature"
    target_dir = "/data7/MEmoBert/evaluation/IEMOCAP/target"

    a = IemocapOriginalDataset(opt, ft_dir, target_dir)
    print(len(a))
    print(next(iter(a)))
    # _iter = iter(a)
    # data1 = next(_iter)
    # data2 = next(_iter)
    # data3 = next(_iter)
    # data1 = a[0]
    # data2 = a[1]
    # data3 = a[2]

    # for k, v in data1.items():
    #     print(k, v.shape)
    # print()
    # for k, v in data2.items():
    #     print(k, v.shape)
    # print()
    # for k, v in data3.items():
    #     print(k, v.shape)
    # print()

    # kk = a.collate_fn([data1, data2, data3])
    # for k, v in kk.items():
    #     print(k, v.shape)
    