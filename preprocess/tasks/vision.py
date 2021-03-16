# import tensorflow as tf
# import collections
# from preprocess.tools.denseface.vision_network.models.dense_net import DenseNet

import os, glob
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from preprocess.utils import get_basename, mkdir
from preprocess.tasks.base_worker import BaseWorker

from preprocess.tools.VAD import VAD
from preprocess.FileOps import read_csv
import math
from code.denseface.config.dense_fer import model_cfg
from code.denseface.model.dense_net import DenseNet
from preprocess.tools.hook import MultiLayerFeatureExtractor

class Video2Frame(BaseWorker):
    def __init__(self, fps=10, save_root='./test', logger=None):
        super().__init__(logger=logger)
        self.fps = fps
        self.save_root = save_root
    
    def __call__(self, video_path):
        basename = get_basename(video_path)
        basename = os.path.join(video_path.split('/')[-2], basename) # video_clip name xxxxx/1.mkv
        save_dir = os.path.join(self.save_root, basename)
        if not(os.path.exists(save_dir) and glob.glob(os.path.join(save_dir, '*.jpg'))):
            mkdir(save_dir)
            cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, self.fps, save_dir) + '%4d.jpg' + " > /dev/null 2>&1"
            os.system(cmd)
            frames_count = len(glob.glob(os.path.join(save_dir, '*.jpg')))
            # self.print('Extract frames from {}, totally {} frames, save to {}'.format(video_path, frames_count, save_dir))
        return save_dir

class Video2FrameTool(BaseWorker):
    def __init__(self, fps=10, logger=None):
        super().__init__(logger=logger)
        self.fps = fps
    
    def __call__(self, video_path, save_dir):
        if not(os.path.exists(save_dir)):
            mkdir(save_dir)
        cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, self.fps, save_dir) + '%4d.jpg' + " > /dev/null 2>&1"
        os.system(cmd)
        frames_count = len(glob.glob(os.path.join(save_dir, '*.jpg')))
        self.print('Extract frames from {}, totally {} frames, save to {}'.format(video_path, frames_count, save_dir))
        return save_dir

class VideoFaceTracker(BaseWorker):
    def __init__(self, openface_dir, save_root='test/track', logger=None):
        super().__init__(logger=logger)
        self.save_root = save_root
        self.openface_dir = openface_dir
    
    def check_exists(self, save_dir):
        clip_num = get_basename(save_dir)
        is_exists = all([
            os.path.exists(os.path.join(save_dir, clip_num+'_aligned')),
            os.path.exists(os.path.join(save_dir, clip_num+'_of_details.txt')),
            os.path.exists(os.path.join(save_dir, clip_num+'.avi')),
            os.path.exists(os.path.join(save_dir, clip_num+'.hog'))
        ])
        return is_exists
    
    def __call__(self, frames_dir):
        basename = get_basename(frames_dir)
        basename = os.path.join(frames_dir.split('/')[-2], basename)
        save_dir = os.path.join(self.save_root, basename)
        if not self.check_exists(save_dir):
            mkdir(save_dir)
            cmd = '{}/FaceLandmarkVidMulti -nomask -fdir {} -out_dir {} > /dev/null 2>&1'.format(
                        self.openface_dir, frames_dir, save_dir
                    )
            os.system(cmd)
            # self.print('Face Track in {}, result save to {}'.format(frames_dir, save_dir))
        return save_dir

class VideoFaceTrackerTool(BaseWorker):
    def __init__(self, openface_dir, logger=None):
        super().__init__(logger=logger)
        self.openface_dir = openface_dir
    
    def check_exists(self, save_dir):
        clip_num = get_basename(save_dir)
        is_exists = all([
            os.path.exists(os.path.join(save_dir, clip_num+'_aligned')),
            os.path.exists(os.path.join(save_dir, clip_num+'_of_details.txt')),
            os.path.exists(os.path.join(save_dir, clip_num+'.avi')),
            os.path.exists(os.path.join(save_dir, clip_num+'.hog'))
        ])
        return is_exists
    
    def __call__(self, frames_dir, save_dir):
        mkdir(save_dir)
        cmd = '{}/FaceLandmarkVidMulti -nomask -fdir {} -out_dir {} > /dev/null 2>&1'.format(
                    self.openface_dir, frames_dir, save_dir
                )
        os.system(cmd)
        self.print('Face Track in {}, result save to {}'.format(frames_dir, save_dir))
        return save_dir

class DensefaceExtractor(BaseWorker):
    def __init__(self, mean=63.987095, std=43.00519, model_path=None, cfg=None, gpu_id=0):
        if cfg is None:
            cfg = model_cfg
        if model_path is None:
            model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
        self.device = torch.device("cuda:{}".format(gpu_id))
        self.extractor = DenseNet(gpu_id, **cfg)
        self.extractor.to(self.device)
        state_dict = torch.load(model_path)
        self.extractor.load_state_dict(state_dict)
        self.extractor.eval()
        self.dim = 342
        self.mean = mean
        self.std = std
        
    def register_midlayer_hook(self, layer_names):
        self.ex_hook = MultiLayerFeatureExtractor(self.extractor, layer_names)
    
    def get_mid_layer_output(self):
        if getattr(self, 'ex_hook') is None:
            raise RuntimeError('Call register_midlayer_hook before calling get_mid_layer_output')
        return self.ex_hook.extract()
    
    def print_network(self):
        self.print(self.extractor)
    
    def __call__(self, img):
        if not isinstance(img, (np.ndarray, str)):
            raise ValueError('Input img parameter must be either str of img path or img np.ndarrays')
        if isinstance(img, np.ndarray):
            if img.shape == (64, 64):
                raise ValueError('Input img ndarray must have shape (64, 64), gray scale img')
        if isinstance(img, str):
            img_path = img
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if not isinstance(img, np.ndarray):
                    raise IOError(f'Warning: Error in {img_path}')
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (64, 64))
        
            else:
                feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
                return feat, np.ones([1, 8]) / 8
            
        # preprocess 
        img = (img - self.mean) / self.std
        img = np.expand_dims(img, -1) # channel = 1
        img = np.expand_dims(img, 0) # batch_size=1

        # forward
        img = torch.from_numpy(img).to(self.device)
        self.extractor.set_input({"images": img})
        self.extractor.forward()
        ft, soft_label = self.extractor.out_ft, self.extractor.pred

        return ft.detach().cpu().numpy(), soft_label.detach().cpu().numpy()


# Deprecated old version
# class DensefaceExtractor(BaseWorker):
#     def __init__(self, restore_path=None, mean=96.3801, std=53.615868, device=0, smooth=False, logger=None):
#         """ extract densenet feature
#             Parameters:
#             ------------------------
#             model: model class returned by function 'load_model'
#         """
#         super().__init__(logger=logger)
#         if restore_path is None:
#             restore_path = '/data2/zjm/tools/FER_models/denseface/DenseNet-BC_growth-rate12_depth100_FERPlus/model/epoch-200'
#         self.model = self.load_model(restore_path)
#         self.mean = mean
#         self.std = std
#         self.previous_img = None        # smooth 的情况下, 如果没有人脸则用上一张人脸填充
#         self.previous_img_path = None
#         self.smooth = smooth
#         self.dim = 342                  # returned feature dim
#         self.device = device
    
#     def load_model(self, restore_path):
#         self.print("Initialize the model..")
#         # fake data_provider
#         growth_rate = 12
#         img_size = 64
#         depth = 100
#         total_blocks = 3
#         reduction = 0.5
#         keep_prob = 1.0
#         bc_mode = True
#         model_path = restore_path
#         dataset = 'FER+'
#         num_class = 8

#         DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
#         data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=num_class)
#         model = DenseNet(data_provider=data_provider, growth_rate=growth_rate, depth=depth,
#                         total_blocks=total_blocks, keep_prob=keep_prob, reduction=reduction,
#                         bc_mode=bc_mode, dataset=dataset)

#         model.saver.restore(model.sess, model_path)
#         self.print("Successfully load model from model path: {}".format(model_path))
#         return model
    
#     def __call__(self, img_path):
#         if os.path.exists(img_path):
#             img = cv2.imread(img_path)
#             if not isinstance(img, np.ndarray):
#                 print(f'Warning: Error in {img_path}')
#                 return None
            
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = cv2.resize(img, (64, 64))
#             if self.smooth:
#                 self.previous_img = img
#                 self.previous_img_path = img_path

#         elif self.smooth and self.previous_img is not None:
#             # print('Path {} does not exists. Use previous img: {}'.format(img_path, self.previous_img_path))
#             img = self.previous_img
        
#         else:
#             feat = np.zeros([1, self.dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
#             return feat
        
#         img = (img - self.mean) / self.std
#         img = np.expand_dims(img, -1) # channel = 1
#         img = np.expand_dims(img, 0) # batch_size=1
#         with tf.device('/gpu:{}'.format(self.device)):
#             feed_dict = {
#                 self.model.images: img,
#                 self.model.is_training: False
#             }

#             # emo index
#             # fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']

#             ft, soft_label = \
#                 self.model.sess.run([self.model.end_points['fc'], 
#                                      self.model.end_points['preds']], feed_dict=feed_dict)
#         return ft, soft_label

class ActiveSpeakerSelector(BaseWorker):
    def __init__(self, diff_threshold=4, logger=None):
        super().__init__(logger=logger)
        self.diff_threshold = diff_threshold
    
    def get_clean_landmark(self, landmark_result_dir):
        '''
        :param alignment_landmark_result_filepath, FaceLandmarkVidMulti 的结果
        set faceId as personId, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,ye]]}, {'frameId':2}]}
        # b[:4] ['frame', ' face_id', ' timestamp', ' confidence']
        # b[299:435] 是 68 点 landmark 的坐标
        '''
        alignment_landmark_result_filepath = os.path.join(landmark_result_dir,
                     get_basename(landmark_result_dir) + ".csv")
        if not os.path.exists(alignment_landmark_result_filepath):
            return None
        
        new_result = {}
        results = read_csv(alignment_landmark_result_filepath, delimiter=',', skip_rows=1) # 去掉 header
        for result in results:
            frameId, faceId, _, confidence = result[:4]
            faceId = eval(faceId)
            raw_landmarks = result[299:435]
            landmarks = []
            for i in range(68):
                landmarks.append([eval(raw_landmarks[i]), eval(raw_landmarks[i+68])])
            assert len(landmarks) == 68
            if new_result.get(faceId) is None:
                new_result[faceId] = []
            new_result[faceId].append({'frameId':eval(frameId), 'score':eval(confidence), 'landmark':landmarks})
        return new_result
    
    def judge_mouth_open(self, landmark):
        '''
        第一个判断条件，嘴是否张开. 如果上下唇的高度大于唇的厚度，那么认为嘴是张开的
        param: 2Dlandmarks of one image, 人脸的2D关键点检测, 68 点的坐标
        '''
        is_open = False
        upperlip=landmark[62][1] - landmark[51][1] # 上唇的厚度
        height=landmark[66][1] - landmark[62][1] # 上下唇的高度
        if height > upperlip:
            is_open = True
        return is_open

    def get_mouth_open_score(self, landmarks):
        '''
        如果某个人的嘴巴张开的次数作为 score, 注意过滤掉confidence小于0.5的脸。
        landmarks of all images of one person.
        '''
        count_open = 0
        count_valid = 0
        for idx in range(len(landmarks)):
            data = landmarks[idx] 
            if data['score'] > 0.5:
                count_valid += 1
                is_open = self.judge_mouth_open(data['landmark'])
                if is_open:
                    count_open += 1
        return count_open, count_valid
    
    def judge_continue_frames_change(self, landmark1, landmark2):
        '''
        连续的两帧的人脸
        根据 62 66 计算内唇高度差，以及 60 64 计算嘴角宽度差，
        根据连续两帧的高度差和宽度差是否发生明显变化来判断.
        :param diff_threshold. = 4 的时候明显的嘴部变化，并且变化帧占总连续帧的25%左右。
        或者 diff_height_threshold > 2 or diff_width_threshold > 2
        '''
        is_change = False
        height1=landmark1[66][1] - landmark1[62][1] # 上下内唇的高度
        width1=landmark1[64][1] - landmark1[60][1] # 最后的内嘴角的宽度
        height2=landmark2[66][1] - landmark2[62][1] # 上下内唇的高度
        width2=landmark2[64][1] - landmark2[60][1] # 最后的内嘴角的宽度
        diff = abs(height2 - height1) + abs(width2 - width1)
        if diff > self.diff_threshold:
            is_change = True
        return is_change

    def get_continue_change_score(self, landmarks):
        '''
        第二个判断条件，嘴是否在动? 统计时间段内哪个人嘴部的动作最多。相对位置的变化，比如上唇和下唇的距离等.
        '''
        count_change = 0
        count_valid = 0
        change_frame_pairs = []
        for idx in range(len(landmarks)-1):
            data1 = landmarks[idx]
            data2 = landmarks[idx+1]
            frameId1 = data1['frameId']
            frameId2 = data2['frameId']
            if frameId2 - frameId1 <= 2:
                if data1['score'] > 0.5 and data2['score'] > 0.5:
                    count_valid += 1
                    is_change = self.judge_continue_frames_change(data1['landmark'], data2['landmark'])
                    if is_change:
                        count_change += 1
                        change_frame_pairs.append([frameId1, frameId2])
        return count_change, count_valid, change_frame_pairs
    
    def get_vad_face_match_score(self, wav_filepath, change_frame_pairs):
        '''
        audio=10ms/frame video=100ms/frame, 将video对齐到audio, 然后计算
        左右vad所在的帧，和 mouth_change 所在的帧, 的重合帧数。
        :param change_frameIds, [[frameId1, frameId2], ..., [frameId100, frameId101]]
        '''
        if len(change_frame_pairs) == 0: # 如果没有动作帧，那么
            return 0, 0
        vad = VAD()
        vad_fts = vad.gen_vad(wav_filepath)     # 获取每一帧的vad的label, 得到vad label序列
        total_frame = len(vad_fts)
        visual_frames = np.zeros_like(vad_fts, dtype=int)     # 构建 video change frames 对齐到 audio frame, 比如某帧
        for frame_pair in change_frame_pairs:
            frameId1, frameId2 = frame_pair
            # frameId start from 1, so frameId=1 到 frameId=2 其实0～100ms
            start_frame = (frameId1-1) * 10
            end_frame = (frameId2-1) * 10
            if end_frame > total_frame:
                end_frame = total_frame
            visual_frames[start_frame: end_frame] = 1
        count_match = 0
        for i in range(total_frame):
            if vad_fts[i] == visual_frames[i] == 1:
                count_match += 1
        return count_match, total_frame
    
    def get_ladder_rank_score(self, raw_person2scores):
        '''
        rank score = 1 / sqrt(rank), ladder的含义是得分相同 的 rank 得分也一样
        raw_person2scores: {0:s1, 1:s2, 2:s3}
        '''
        sort_person2score_tups = sorted(raw_person2scores.items(), key=lambda asd: asd[1], reverse=True)
        rank_ind = 1
        person2rank_score = {}
        previous_score = sort_person2score_tups[0][1]
        for tup in sort_person2score_tups:
            cur_person, cur_score = tup
            if cur_score == previous_score:
                rank_score = 1 / math.sqrt(rank_ind)
                rank_ind += 0
            else:
                rank_ind += 1
                rank_score = 1 / math.sqrt(rank_ind)
            person2rank_score[cur_person] = rank_score
            previous_score = cur_score
        return person2rank_score
    
    def get_final_decision(self, open_person2scores, change_person2scores, 
                match_person2scores):
        '''
        条件1: 如果三个得分中有一个是0, 即如果张嘴次数是0 或者 动作次数是0 或者 match次数是0，那么该人不是说话人。
        条件2: 如果只剩下一个人, 那么直接该人的ID. 
        条件3: 如果最后没有合适的人，那么该视频丢弃. return None
        条件4: 如果正常的多个人，那么进行排序
        open_person2scores: {p1:s1, p2:s2, p3:s3}
        '''
        # 条件1
        persons = list(open_person2scores.keys())
        for person in persons:
            if open_person2scores[person] == 0 or change_person2scores[person] == 0 \
                    or match_person2scores[person] == 0:
                open_person2scores.pop(person)
                change_person2scores.pop(person)
                match_person2scores.pop(person)
        # 条件2 & 条件3
        if len(open_person2scores) == 0:
            return None
        if len(open_person2scores) == 1:
            return list(open_person2scores.keys())[0]
        # 条件4
        open_rank_scores = self.get_ladder_rank_score(open_person2scores)
        change_rank_scores = self.get_ladder_rank_score(change_person2scores)
        match_rank_scores = self.get_ladder_rank_score(match_person2scores)

        person2final_score = {}
        for person in open_rank_scores.keys():
            final_score = open_rank_scores[person] + change_rank_scores[person] + match_rank_scores[person]
            person2final_score[person] = final_score
        sort_person2score_tups = sorted(person2final_score.items(), key=lambda asd: asd[1], reverse=True)
        return sort_person2score_tups[0][0]

    def __call__(self, face_dir, audio_path):
        '''
        通过 openface 的结果判断一个人是否在说话，很有可能不是屏幕中的人在说话。
        Q1: 得到次数之后如何计算得分？ rank_score = 1 / sqrt(rank)
        Q2: 三个得分的融合决策的策略是什么？ rank_score 直接相加。
        :param, landmarks_path, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,y2]]}, {'frameId':2}]}
        '''
        save_path = os.path.join(face_dir, 'has_active_spk.txt')
        # if os.path.exists(save_path):
        #     content = open(save_path).read().strip()
        #     return None if content == 'None' else int(content)
        
        if not os.path.exists(audio_path):
            # print('[Debug] There is no wav info and return None!')
            f = open(save_path, 'w')
            f.write("None")
            return None
        
        landmarks = self.get_clean_landmark(face_dir)
        if landmarks is None:
            # print('[Debug] There is no landmarks and return None!')
            f = open(save_path, 'w')
            f.write("None")
            return None
        
        open_person2scores = {}
        change_person2scores = {}
        match_person2scores = {}
        for person in landmarks.keys():
            person_frames = landmarks[person]
            count_open, count_valid = self.get_mouth_open_score(person_frames)
            # self.print('[Debug]open person{} open: {}/{}'.format(person, count_open, count_valid))
            count_change, count_valid, change_frame_pairs = self.get_continue_change_score(person_frames)
            # self.print('[Debug]continue person{} change: {}/{}'.format(person, count_change, count_valid))
            # self.print('[Debug]change_frame_pairs {}'.format(change_frame_pairs))
            count_match, total_frame = self.get_vad_face_match_score(audio_path, change_frame_pairs)
            # self.print('[Debug]vad person{} match: {}/{}'.format(person, count_match, total_frame))
            open_person2scores[person] = count_open
            change_person2scores[person] = count_change
            match_person2scores[person] = count_match

        active_speaker = self.get_final_decision(open_person2scores, change_person2scores, match_person2scores)
        f = open(save_path, 'w')
        f.write(str(active_speaker))
        return active_speaker
    
class FaceSelector(BaseWorker):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
    
    def __call__(self, face_dir, active_spk_id):
        if active_spk_id == None:
            return []
        basename = get_basename(face_dir)
        face_img_dir = os.path.join(face_dir, basename + '_aligned')
        face_list = glob.glob(os.path.join(face_img_dir, f'*_det_{active_spk_id:02d}_*.bmp'))
        face_list = sorted(face_list, key=lambda x:int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        data_frame = pd.read_csv(os.path.join(face_dir, basename+'.csv'))
        data_frame = data_frame[data_frame[' face_id']==active_spk_id]
        ans = []
        for face_img in face_list:
            frame_num = int(face_img.split('/')[-1].split('.')[0].split('_')[-1])
            row = data_frame[data_frame['frame']==frame_num]
            confidence = float(row[' confidence']) if len(row) else 0
            ans.append(
                {
                    'img': face_img,
                    'confidence': confidence,
                    'frame_num': frame_num
                }
            )
        return ans

if __name__ == '__main__':
    # get_frame = Video2Frame()
    # frame_dir = get_frame('../resources/output1.mkv')
    # face_track = VideoFaceTracker()
    # a = face_track(frame_dir)
    # print(a)
    # face_path = '/data6/zjm/emobert/preprocess/test/track/output1/output1_aligned/frame_det_00_000010.bmp'
    # mean = 96.3801
    # std = 53.615868
    # get_denseface = DensefaceExtractor(mean=mean, std=std, device=0)
    # feature = get_denseface(face_path)
    # print(feature.shape)
    # import time
    select_activate_spk = ActiveSpeakerSelector()
    # # select_faces = FaceSelector()
    # start = time.time()
    active_spkid = select_activate_spk("/data7/emobert/data_nomask_new/faces/No0030.About.Time.Error/4", "/data7/emobert/data_nomask_new/audio_clips/No0030.About.Time.Error/4.wav")
    print(active_spkid)
    # # face_lists = select_faces("test/track/output1", active_spkid)
    # end = time.time()
    # print(end-start)
    # print(active_spkid)
    # # print(face_lists)
    # img = '/data7/MEmoBert/preprocess/data/faces/No0009.The.Truman.Show/42/42_aligned/frame_det_00_000038.bmp'
    # img = '/data7/MEmoBert/preprocess/data/frames/No0009.The.Truman.Show/42/0038.jpg'
    
    # frame_dir = 'data/frames/No0009.The.Truman.Show/42'
    # face_track = VideoFaceTracker(save_root='mask')
    # a = face_track(frame_dir)
    # print(a)

    # img = '/data7/MEmoBert/preprocess/test_track/No0009.The.Truman.Show/42/42_aligned/frame_det_00_000020.bmp'
    # detector = Detector('/root/tools/insightface/RetinaFace/pretrained', 7)
    # bbox, landmark = detector.detect(img)
    # print(bbox)
    # detector.draw_detection(img)

    # a = FaceSelector()
    # ans = a('/data7/MEmoBert/preprocess/data/faces/No0011.American.Beauty/20', 0)
    # print(ans)
   
    # a = DensefaceExtractor()
    # a.register_midlayer_hook([
    #     "features.transition1.relu",
    #     "features.transition2.relu"
    # ])
    # img = '/data7/MEmoBert/preprocess/data/faces/No0007.Schindler.List/19/19_aligned/frame_det_01_000009.bmp'
    # ft, pred = a(img)
    # print(ft.shape, pred.shape)
    # trans1, trans2 = a.get_mid_layer_output()
    # print(trans1.shape)
    # print(trans2.shape)