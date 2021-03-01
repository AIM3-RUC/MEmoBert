from apex.amp.frontend import state_dict
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, '/data7/MEmoBert/')
from code.denseface.model.dense_net import DenseNet, DenseNetEncoder
from code.denseface.model.vggnet import VggNet, VggNetEncoder
from code.denseface.model.resnet import ResNet, ResNetEncoder
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
    model_type = 'resnet'
    if model_type == 'vggnet':
        print("--- Use Vggnet")
        from code.denseface.config.vgg_fer import model_cfg
    elif model_type == 'densenet':
        print("--- Use Densenet")
        from code.denseface.config.dense_fer import model_cfg
    elif model_type == 'resnet':
        from code.denseface.config.res_fer import model_cfg
    else:
        print("[Error] model type {}".format(model_type))
        exit(0)
    imgs_path = '/data3/zjm/dataset/ferplus/npy_data/val_img.npy'
    image_index = 2
    imgs = np.load(imgs_path)
    print('{} imgs'.format(len(imgs)))
    cv2.imwrite('./pics/val_img{}.jpg'.format(image_index), imgs[image_index])

    image = np.expand_dims(imgs[image_index], 2)
    image = normalize_image_by_chanel(image)
    print(image.shape)

    device = torch.device('cuda:0')
    if 'vgg' in model_cfg['model_name']:
        model_path = "/data7/MEmoBert/emobert/exp/face_model/vggnet_adam0.0001_0.25/ckpts/model_step_68.pt"
        extractor = VggNetEncoder(**model_cfg)
    elif 'dense' in model_cfg['model_name']:
        extractor = DenseNetEncoder(**model_cfg)
        model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
    elif 'res' in model_cfg['model_name']:
        extractor = ResNetEncoder(**model_cfg)
        model_path = "/data7/MEmoBert/emobert/exp/face_model/resnet18_adam_warmup_run2_adam0.0005_0.0/ckpts/model_step_68.pt"
    else:
        extractor =None
    print(extractor)
    state_dict = torch.load(model_path)
    for key in list(state_dict.keys()):
        if 'classifier' in key:
            del state_dict[key]
    extractor.load_state_dict(state_dict)
    extractor.eval()
    extractor.to(device)

    ### for densenet
    # select_layers = ['features.conv0', 
    #     'features.denseblock1.denselayer16.conv1',
    #     'features.transition1.relu',
    #     'features.denseblock2.denselayer16.conv1',
    #     'features.transition2.relu',
    #     'features.denseblock3.denselayer16.conv1',
    # ]
    ### for vggnet
    # select_layers = [
    #     'conv_block1.conv2',
    #     'conv_block1.relu2',
    #     'conv_block2.conv2',
    #     'conv_block3.conv2',
    #     'conv_block4.conv2',
    # ]
    ### for resnet
    select_layers = [
        'features.pool0',
        'features.resblock1.0.conv2',
        'features.resblock2.0.conv2',
    ]
    ex = MultiLayerFeatureExtractor(extractor, select_layers)
    x = torch.FloatTensor([image]).to(device)
    x = x.squeeze(-1)  ## torch.Size([1, 64, 64])
    extractor.forward(x) 
    outputs = ex.extract()
    assert len(outputs) == len(select_layers)
    for i in range(len(select_layers)):
        print('save {} {}'.format(select_layers[i], outputs[i].shape))
        save_feature_to_img(outputs[i], image_name='val_img{}'.format(image_index), layer_name=select_layers[i])