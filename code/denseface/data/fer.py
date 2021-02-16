import torch
import torch.utils.data as data

import cv2
import random
import numpy as np
from os.path import join
from .base_provider import ImagesDataSet

def augment_image(image, pad=8):
    '''
    Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally
    '''
    init_shape = image.shape
    img_size = init_shape[0]
    new_shape = [init_shape[0] + pad * 2, init_shape[1] + pad * 2, init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    ## randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[init_x: init_x + init_shape[0],
                    init_y: init_y + init_shape[1], :]
    ## randomly flip
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1, :]
    # randomly rotation
    angle = np.random.randint(-15, 16)
    rot_mat = cv2.getRotationMatrix2D((img_size, img_size), angle, 1.)
    cropped = cv2.warpAffine(cropped, rot_mat, (img_size, img_size))
    if len(cropped.shape) == 2:
        cropped = np.expand_dims(cropped, 2)
    return cropped

def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=pad)
    return new_images

class FERPlusDataSet(ImagesDataSet):
    def __init__(self, config, data_dir, target_dir, setname='trn'):
        ''' 
        Fer plus dataset used for read one image and its transform
        '''
        super().__init__()
        self.config = config
        image_path = join(data_dir, '{}_img.npy'.format(setname))
        target_path = join(target_dir, '{}_target.npy'.format(setname))
        self.images = np.expand_dims(np.load(image_path), 3)
        print('Images {}'.format(self.images.shape))
        self.label = np.load(target_path)
        # 归一化～
        if self.config.normalization is not None:
            self.images = self.normalize_images(self.images, self.config.normalization)

        if len(self.label.shape) > 1:
            self.label = np.argmax(self.label, axis=1)            
        self.manual_collate_fn = True

    def __getitem__(self, index):
        example = {}
        image = self.images[index]
        if self.config.data_augmentation:
            image = augment_image(image, pad=8)
        image = torch.tensor(image)
        label = torch.tensor(self.label[index])
        example['image'] = image
        example['label'] = label
        return example
    
    @property
    def data_shape(self):
        return (64, 64, 1)

    def __len__(self):
        return len(self.label)
           
    def collate_fn(self, batch):
        ret = {}
        images = [sample['image'].numpy() for sample in batch]
        label = [sample['label'] for sample in batch]
        label = torch.tensor(label)
        images = torch.tensor(images)
        ret["labels"] = label
        ret["images"] = images
        return ret

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def __init__(self, opt, data_dir, target_dir, setname='trn', is_train=True, **kwargs):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.dataset = FERPlusDataSet(opt, data_dir, target_dir, setname, **kwargs)
        
        ''' Whether to use manual collate function defined in dataset.collate_fn'''
        if self.dataset.manual_collate_fn: 
            print('Use the self batch collection methods')
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=is_train,
                num_workers=int(opt.num_threads),
                drop_last=is_train,
                collate_fn=self.dataset.collate_fn
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=is_train,
                num_workers=int(opt.num_threads),
                drop_last=is_train
            )

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

if __name__ == '__main__':
    data_path = '/data3/zjm/dataset/ferplus/npy_data'
    target_path = '/data3/zjm/dataset/ferplus/npy_data'
    fer_dataloader = CustomDatasetDataLoader(config, data_path, target_path, setname='val')
    print(fer_dataloader.__len__)