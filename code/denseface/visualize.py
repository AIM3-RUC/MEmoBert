from apex.amp.frontend import state_dict
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, '/data7/MEmoBert/')
from code.denseface.model.dense_net import DenseNet, DenseNetEncoder
from code.denseface.config.conf_fer import model_cfg
from code.denseface.hook_demo import MultiLayerFeatureExtractor

def normalize_image_by_chanel(image):
    new_image = np.zeros(image.shape)
    for chanel in range(image.shape[-1]):
        mean = np.mean(image[:, :, chanel])
        std = np.std(image[:, :, chanel])
        new_image[:, :, chanel] = (image[:, :, chanel] - mean) / std
    return new_image

def save_feature_to_img(feature, feature_map_ind=0, image_name='val_img3', layer_name='features.conv0'):
    #to numpy
    feature=feature[0].detach().cpu().numpy()
    #use sigmod to [0,1]
    feature= 1.0/(1+np.exp(-1*feature))
    # to [0, 255]
    feature=np.round(feature*255)
    cv2.imwrite('./pics/{}_{}.jpg'.format(image_name, layer_name), feature[feature_map_ind])

if __name__ == '__main__':
    imgs_path = '/data3/zjm/dataset/ferplus/npy_data/val_img.npy'
    image_index = 2
    imgs = np.load(imgs_path)
    print('{} imgs'.format(len(imgs)))
    cv2.imwrite('./pics/val_img{}.jpg'.format(image_index), imgs[image_index])

    image = np.expand_dims(imgs[image_index], 2)
    image = normalize_image_by_chanel(image)
    print(image.shape)

    device = torch.device('cuda:0')
    extractor = DenseNetEncoder(**model_cfg)
    print(extractor)
    model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
    state_dict = torch.load(model_path)
    for key in list(state_dict.keys()):
        if 'classifier' in key:
            del state_dict[key]
    extractor.load_state_dict(state_dict)
    extractor.eval()
    extractor.to(device)

    select_layers = ['features.conv0', 
        'features.denseblock1.denselayer16.conv1',
        'features.transition1.relu',
        'features.denseblock2.denselayer16.conv1',
        'features.transition2.relu',
        'features.denseblock3.denselayer16.conv1',
    ]
    ex = MultiLayerFeatureExtractor(extractor, select_layers)
    x = torch.FloatTensor([image]).to(device)
    x = x.squeeze(-1)
    print(x.shape)
    extractor.forward(x)
    conv0, b1l1conv1, trans1, b1l1conv2, trans2, b2l2conv1 = ex.extract()
    print(conv0.shape)
    print(b1l1conv1.shape)
    print('tans1 {}'.format(trans1.shape))
    # out_tans1_ft = F.avg_pool2d(trans1, kernel_size=32, stride=1).view(trans1.size(0), -1)  # torch.Size([64, 216])
    print(b1l1conv2.shape)
    print('tans2 {}'.format(trans2.shape))
    # out_tans2_ft = F.avg_pool2d(trans2, kernel_size=16, stride=1).view(trans2.size(0), -1)  # torch.Size([64, 300])
    print(b2l2conv1.shape)
    save_feature_to_img(conv0, image_name='val_img{}'.format(image_index), layer_name=select_layers[0])
    save_feature_to_img(b1l1conv1, image_name='val_img{}'.format(image_index), layer_name=select_layers[1])
    save_feature_to_img(trans1, image_name='val_img{}'.format(image_index), layer_name=select_layers[2])
    save_feature_to_img(b1l1conv2, image_name='val_img{}'.format(image_index), layer_name=select_layers[3])
    save_feature_to_img(trans2, image_name='val_img{}'.format(image_index), layer_name=select_layers[4])
    save_feature_to_img(b2l2conv1, image_name='val_img{}'.format(image_index), layer_name=select_layers[5])
