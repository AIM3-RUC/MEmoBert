import os, sys, glob
import cv2
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import traceback

sys.path.append('/data7/lrc/MuSe2020/MuSe2020_features/code/denseface/vision_network')
from models.dense_net import DenseNet

def load_model(restore_path):
    print("Initialize the model..")
    # fake data_provider
    growth_rate = 12
    img_size = 64
    depth = 100
    total_blocks = 3
    reduction = 0.5
    keep_prob = 1.0
    bc_mode = True
    model_path = restore_path
    dataset = 'MUSE'
    num_class = 2

    DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
    data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=num_class)
    model = DenseNet(data_provider=data_provider, growth_rate=growth_rate, depth=depth,
                     total_blocks=total_blocks, keep_prob=keep_prob, reduction=reduction,
                     bc_mode=bc_mode, dataset=dataset)

    end_points = model.end_points
    model.saver.restore(model.sess, model_path)
    print("Successfully load model from model path: {}".format(model_path))
    return model

class FeatureExtractor(object):
    def __init__(self, model, mean, std, smooth=True):
        """ extract densenet feature
            Parameters:
            ------------------------
            model: model class returned by function 'load_model'
        """
        self.model = model
        self.mean = mean
        self.std = std
        self.previous_img = None        # smooth 的情况下, 如果没有人脸则用上一张人脸填充
        self.previous_img_path = None
        self.smooth = smooth
        self.dim = 342                  # returned feature dim
    
    def extract_feature(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            if self.smooth:
                self.previous_img = img
                self.previous_img_path = img_path

        elif self.smooth and self.previous_img is not None:
            # print('Path {} does not exists. Use previous img: {}'.format(img_path, self.previous_img_path))
            img = self.previous_img
        
        else:
            feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
            return feat
        
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, 3) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1
        feed_dict = {
            self.model.images: img,
            self.model.is_training: False
        }
        ft = self.model.sess.run(self.model.end_points['fc'], feed_dict=feed_dict)
        return ft

def get_model_path(key):
    return {
        'csz': '/data2/zjm/AVEC2019/CES/feature/visual/model.denseface.k12.tune_csz/finetuned.model/finetune.train/DenseNet-BC_growth_rate=12_depth=100_dataset_AVEC/model/epoch-6',
        'zjm': '/data2/zjm/AVEC2019/CES/feature/visual/model.denseface.k12.tune_zjm_de_hu_trn_run3/finetune.train.valence/DenseNet-BC_growth_rate=12_depth=100_dataset_AVEC/model/epoch-15', 
        'lrc': '/data7/lrc/MuSe2020/MuSe2020_features/code/wild/finetune_model/denseface_lrc_tune_av/DenseNet-BC_growth_rate=12_depth=100_dataset_MUSE/model/epoch-23',
        'lrc_fix': '/data7/lrc/MuSe2020/MuSe2020_features/code/wild/finetune_model/denseface_lrc_tune_av_fix/DenseNet-BC_growth_rate=12_depth=100_dataset_MUSE/model/epoch-12',
        'lrc_aug': '/data7/lrc/MuSe2020/MuSe2020_features/code/wild/finetune_model/denseface_lrc_tune_av_fix_aug/DenseNet-BC_growth_rate=12_depth=100_dataset_MUSE/model/epoch-10'
    }[key]

def read_timestamp(csv_path):
    ''' 读取timestamp, 读取路径为:
        /data7/lrc/MuSe2020/MuSe2020_raw/labels/data/processed_tasks/c1_muse_wild/label_segments/arousal
    '''
    data = pd.read_csv(csv_path)
    timestamp = data['timestamp']
    return np.array(timestamp)

def make_denseface(model_name, start, end): 
    mean = 96.3801
    std = 53.615868
    smooth = False
    restore_path = get_model_path(model_name)
    model = load_model(restore_path)
    extractor = FeatureExtractor(model, mean=mean, std=std, smooth=smooth)
    for videoid in os.listdir(face_root)[start:end]:
        try:
            frame_dir = os.path.join(frame_root, videoid)
            face_dir = os.path.join(face_root, videoid)
            if not os.path.exists(face_dir):
                raise RuntimeError('Face should be extract first for video:{}'.format(videoid))

            save_dir = os.path.join(save_root, videoid)
            ans = []
            frame_pics = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
            for frame_pic in tqdm(frame_pics):
                basename = os.path.basename(frame_pic)
                face_path = os.path.join(face_dir, basename)
                feat = extractor.extract_feature(face_path)
                ans.append(feat)
            ans = np.concatenate(ans)
            print(videoid, ans.shape)
            save_path = os.path.join(save_root, videoid + ".npy")
            np.save(save_path, ans)
        except BaseException as e:
            log = open(f'log/denseface/{videoid}.log', 'w')
            log.write(traceback.format_exc())
            raise RuntimeError(traceback.format_exc())

if __name__ == '__main__':
    import sys
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    print(f"Start {start} end {end}")
    face_root = '../Face'
    frame_root = '../Frame'
    save_root = '../../MSP-IMPROV_feature/face/raw'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    make_denseface('csz', start, end)
    