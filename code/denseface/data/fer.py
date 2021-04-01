import torch
from torch._C import default_generator
import torch.utils.data as data

import cv2
import random
import numpy as np
from os.path import join
from .base_provider import ImagesDataSet
from torchvision import transforms
from PIL import Image

def augment_image(image, pad=8):
    '''
    input shape = [img-size, img-size, channels]
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
    cropped = zeros_padded[init_x: init_x + init_shape[0], init_y: init_y + init_shape[1], :]
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

def augment_image2d(image, pad=8):
    '''
    2D input shape = [img-size, img-size]
    Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally
    '''
    init_shape = image.shape
    img_size = init_shape[0]
    new_shape = [init_shape[0] + pad * 2, init_shape[1] + pad * 2]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad] = image
    ## randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[init_x: init_x + init_shape[0], init_y: init_y + init_shape[1]]
    ## randomly flip
    flip = random.getrandbits(1)
    if flip:
        cropped = cropped[:, ::-1]
    # randomly rotation
    angle = np.random.randint(-15, 16)
    rot_mat = cv2.getRotationMatrix2D((img_size, img_size), angle, 1.)
    cropped = cv2.warpAffine(cropped, rot_mat, (img_size, img_size))
    return cropped

def augment_image_compose(img, image_size, pad):
    # with input [img-size, img-size] 
    # Note: Must based on the unit8, can't handel the norm images.
    # image_size: 64
    # pad: 8
    compose_aug = transforms.Compose([
            # pad on four edge and random crop
            transforms.RandomCrop(image_size, padding=pad),
            # 50 prob to flip or not
            transforms.RandomHorizontalFlip(p=0.5),
            # random rotation
            transforms.RandomRotation((-15, 16))
        ])
    # print(img.shape, type(img), type(img[0][0]))
    aug_img = compose_aug(Image.fromarray(img))
    return np.array(aug_img)

def augment_batch_images(initial_images, pad=8, image_size=112):
    # for uniter
    # input shape = [batch, img-size, img-size]
    # return = [batch, img-size, img-size]
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        # default is float 16, not work. float32 is ok
        new_images[i] = augment_image_compose(initial_images[i].astype('float32'), pad=pad, image_size=image_size)
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