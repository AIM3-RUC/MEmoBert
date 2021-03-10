from torchvision import transforms
import torch
import cv2
import numpy as np
import sys
from PIL import Image
from data.fer import augment_image2d

imgs_path = '/data3/zjm/dataset/ferplus/npy_data/val_img.npy'
image_index = 2
imgs = np.load(imgs_path)
print('{} imgs'.format(len(imgs)))
img = imgs[image_index]
cv2.imwrite('./pics/augment/val_img{}.jpg'.format(image_index), img)

ori_aug_img =  augment_image2d(img)
cv2.imwrite('./pics/augment/val_img{}_ori_aug.jpg'.format(image_index), ori_aug_img)

mean, std = 63, 43
img = (img - mean) / std
print(img)
print(type(img[0][0]))
compose_aug = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation((-15, 16))
])
aug_img = compose_aug(Image.fromarray(img))
print(np.array(aug_img).shape)
cv2.imwrite('./pics/augment/val_img{}_compose_aug.jpg'.format(image_index), np.array(aug_img))

if False:
    # pad and randomcrop
    # crop_aug = transforms.Compose(transforms.RandomCrop(64, padding=(8,8,8,8)))
    crop_aug = transforms.RandomCrop(64, padding=8)
    crop_aug_img = crop_aug(Image.fromarray(ori_aug_img))
    cv2.imwrite('./pics/augment/val_img{}_crop_aug.jpg'.format(image_index), np.array(crop_aug_img))

    ## rand flip 翻转
    flip_aug = transforms.RandomHorizontalFlip(p=0.5)
    flip_aug_img = flip_aug(crop_aug_img)
    cv2.imwrite('./pics/augment/val_img{}_crop_flip_aug.jpg'.format(image_index), np.array(flip_aug_img))

    ## randomly rotation -15 16 ang
    rota_aug = transforms.RandomRotation((-15, 16))
    rota_aug_img = flip_aug(flip_aug_img)
    cv2.imwrite('./pics/augment/val_img{}_crop_flip_rota_aug.jpg'.format(image_index), np.array(rota_aug_img))










