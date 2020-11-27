import os
import numpy as np
import math
from emobert.preprocess.tools.VAD import VAD
from emobert.preprocess.FileOps import read_csv, write_pkl, read_pkl

'''
验证采用landmark 和 FAU 的规则方法进行说话人的追踪. 设定合适的规则进行筛选。
https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments
Step1: 进行 单脸/多脸 landmark 的可视化 --done
Step2: 进行视频的 landmark 追踪效果的可视化 --done
分别判断多个人的嘴巴问题，可以
Step3: 综合三个判断条件
'''

def get_frames_from_video(video_path, fps, save_dir):
    '''
    param fps: how many frames per second
    param save_dir: output_frames_dir
    '''
    cmd = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/'.format(video_path, fps, save_dir) + '%4d.jpg'
    os.system(cmd)

def get_landmark_for_image(tool_dir, img_path, save_dir):
    '''
    使用 FaceLandmarkImg 工具
    https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format#facelandmarkimg
    '''
    cmd = '{}/FaceLandmarkImg -f {} -out_dir {}'.format(tool_dir, img_path, save_dir)
    os.system(cmd)

def get_landmark_for_video(tool_dir, video_path, save_dir):
    '''
    首先需要将video转化为frames.
    使用 FaceLandmarkVid 工具
    '''
    cmd = '{}/FaceLandmarkVidMulti -fdir {} -mask -out_dir {}'.format(tool_dir, video_path, save_dir)
    os.system(cmd)

def judge_mouth_open(landmark):
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

def get_mouth_open_score(landmarks):
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
            is_open = judge_mouth_open(data['landmark'])
            if is_open:
                count_open += 1
    return count_open, count_valid

def judge_continue_frames_change(landmark1, landmark2, diff_threshold=4):
    '''
    连续的两帧的人脸，相对像素差。
    根据 62 66 计算内唇高度差，以及 60 64 计算嘴角宽度差，
    根据连续两帧的高度差和宽度差是否发生明显变化来判断.
    :param diff_threshold. = 4 的时候明显的嘴部变化，并且变化帧占总连续帧的 25% 左右。
    或者 diff_height_threshold > 2 or diff_width_threshold > 2
    '''
    is_change = False
    height1=landmark1[66][1] - landmark1[62][1] # 上下内唇的高度
    width1=landmark1[64][1] - landmark1[60][1] # 最后的内嘴角的宽度
    height2=landmark2[66][1] - landmark2[62][1] # 上下内唇的高度
    width2=landmark2[64][1] - landmark2[60][1] # 最后的内嘴角的宽度
    # print(landmark1[66][1], landmark1[62][1])
    # print(landmark2[66][1], landmark2[62][1])
    # print('height {} {} and width {} {}'.format(height1, height2, width1, width2))
    diff = abs(height2 - height1) + abs(width2 - width1)
    # print(diff)
    if diff > diff_threshold:
        is_change = True
    return is_change

def get_continue_change_score(landmarks):
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
        # print(frameId1, frameId2)
        # 当相邻的两帧的ID小于2的时候
        if frameId2 - frameId1 <= 2:
            if data1['score'] > 0.5 and data2['score'] > 0.5:
                count_valid += 1
                is_change = judge_continue_frames_change(data1['landmark'], data2['landmark'])
                if is_change:
                    count_change += 1
                    change_frame_pairs.append([frameId1, frameId2])
    return count_change, count_valid, change_frame_pairs

def get_vad_face_match_score(wav_filepath, change_frame_pairs):
    '''
    audio=10ms/frame video=100ms/frame, 将video对齐到audio, 然后计算
    左右vad所在的帧，和 mouth_change 所在的帧, 的重合帧数。
    :param change_frameIds, [[frameId1, frameId2], ..., [frameId100, frameId101]]
    '''
    if len(change_frame_pairs) == 0: # 如果没有动作帧，那么
        return 0, 0
    vad = VAD()
    vad_fts = vad.gen_vad(audio_path)     # 获取每一帧的vad的label, 得到vad label序列
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

def get_clean_landmarks_results(alignment_landmark_result_filepath, clean_result_save_path):
    '''
    :param alignment_landmark_result_filepath, FaceLandmarkVidMulti 的结果
    set faceId as personId, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,ye]]}, {'frameId':2}]}
    # b[:4] ['frame', ' face_id', ' timestamp', ' confidence']
    # b[299:435] 是 68 点 landmark 的坐标
    '''
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
    write_pkl(clean_result_save_path, new_result)


def get_ladder_rank_score(raw_person2scores):
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

def get_final_decision(open_person2scores, change_person2scores, match_person2scores):
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
        if open_person2scores[person] == 0 or change_person2scores[person] == 0 or match_person2scores[person] == 0:
            open_person2scores.pop(person)
            change_person2scores.pop(person)
            match_person2scores.pop(person)
    # 条件2 & 条件3
    if len(open_person2scores) == 0:
        return None
    if len(open_person2scores) == 1:
        return list(open_person2scores.keys())[0]
    # 条件4
    open_rank_scores = get_ladder_rank_score(open_person2scores)
    change_rank_scores = get_ladder_rank_score(change_person2scores)
    match_rank_scores = get_ladder_rank_score(match_person2scores)
    # print(open_rank_scores)
    # print(change_rank_scores)
    # print(match_rank_scores)
    person2final_score = {}
    for person in open_rank_scores.keys():
        final_score = open_rank_scores[person] + change_rank_scores[person] + match_rank_scores[person]
        person2final_score[person] = final_score
    sort_person2score_tups = sorted(person2final_score.items(), key=lambda asd: asd[1], reverse=True)
    return sort_person2score_tups[0][0]

def judge_active_speaker(landmark_filepath, wav_filepath):
    '''
    通过 openface 的结果判断一个人是否在说话，很有可能不是屏幕中的人在说话。
    Q1: 得到次数之后如何计算得分？ rank_score = 1 / sqrt(rank)
    Q2: 三个得分的融合决策的策略是什么？ rank_score 直接相加。
    :param, landmarks_path, {'p1': [{'frameId':1, 'score':0.99, 'landmark':[[x1,y1], [x2,ye]]}, {'frameId':2}]}
    '''
    landmarks = read_pkl(landmark_filepath)
    open_person2scores = {}
    change_person2scores = {}
    match_person2scores = {}
    for person in landmarks.keys():
        person_frames = landmarks[person]
        count_open, count_valid = get_mouth_open_score(person_frames)
        print('person{} open: {}/{}'.format(person, count_open, count_valid))
        count_change, count_valid, change_frame_pairs = get_continue_change_score(person_frames)
        print('person{} change: {}/{}'.format(person, count_change, count_valid))
        count_match, total_frame = get_vad_face_match_score(wav_filepath, change_frame_pairs)
        print('person{} match: {}/{}'.format(person, count_match, total_frame))
        open_person2scores[person] = count_open
        change_person2scores[person] = count_change
        match_person2scores[person] = count_match
    # print(open_person2scores)
    # print(change_person2scores)
    # print(match_person2scores)
    active_speaker = get_final_decision(open_person2scores, change_person2scores, match_person2scores)
    return active_speaker

if __name__ == "__main__":
    tool_dir = '/root/tools/openface_tool/OpenFace/build/bin'
    # 34:30 34:50 No0001
    video_path = '../resources/output2.avi'
    audio_path = '../../resources/output2.wav'
    output_frames_dir = '../resources/output2_frames'
    output_dir = './asv_exp'

    if False:
        if not os.path.exists(output_frames_dir):
            os.mkdir(output_frames_dir)
        get_frames_from_video(video_path, 10, output_frames_dir)
    
    if False:
        output_dir = os.path.join(output_dir, 'single_image')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        img_path = os.path.join(output_frames_dir, '0001.jpg')
        get_landmark_for_image(tool_dir, img_path, output_dir)
    
    if False:
        output_dir = os.path.join(output_dir, 'video2')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        get_landmark_for_video(tool_dir, output_frames_dir, output_dir)
    
    if False:
        raw_landmark_filepath = os.path.join(output_dir, 'video2/output2_frames.csv')
        landmark_filepath = os.path.join(output_dir, 'video2/landmarks.pkl')
        get_clean_landmarks_results(raw_landmark_filepath, landmark_filepath)
    
    if True:
        # 文档中三个条件进行判断
        landmark_filepath = os.path.join(output_dir, 'video2/landmarks.pkl')
        active_speker = judge_active_speaker(landmark_filepath, audio_path)
        print('active_speker : {}'.format(active_speker))