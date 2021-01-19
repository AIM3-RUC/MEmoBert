from __future__ import print_function
import os, glob
import numpy as np
import cv2
import sys
import multiprocessing
import time
import traceback

def extract_one_video(video_path, output_dir):
    if not os.path.exists(video_path):
        raise ValueError(f'{video_path} not exists')

    # frame_sample_rate = 3 # IEMOCAP是帧全抽, 脸3帧里抽一帧, 但是帧全抽太占硬盘, 这里抽帧的时候就3帧一抽了
    frame_sample_rate = 6   # 我们希望脸的fps=10, MSP原始视频的fps=60, 所以这里6帧里抽一帧 
    video_capture = cv2.VideoCapture(video_path)
    kth_frame = 0
    n_extract = 0
    success, frame = video_capture.read()
    while success:
        if kth_frame % frame_sample_rate == 0:
            cv2.imwrite(os.path.join(output_dir, '{:05d}.jpg'.format(kth_frame)), frame)
            n_extract += 1
            # print('{:05d}.jpg'.format(kth_frame))
        
        kth_frame += 1
        success, frame = video_capture.read()  # next frame
    print('{} total frame:{} extracted:{}'.format(video_path, kth_frame, n_extract))

def get_face(frame_dir):
    name = multiprocessing.current_process().name
    source_lst = f'.tmp/source.lst.{name}'
    target_lst = f'.tmp/target.lst.{name}'
    frame_jpgs = glob.glob(os.path.join(frame_dir, '*.jpg'))
    frame_jpgs = list(map(lambda x: x+'\n', frame_jpgs))
    output_jpgs = list(map(lambda x: x.replace('Frame', 'Face'), frame_jpgs))
    if len(output_jpgs) == 0:
        return 

    output_dir = os.path.dirname(output_jpgs[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(source_lst, 'w') as f:
        f.writelines(frame_jpgs)
    with open(target_lst, 'w') as f:
        f.writelines(output_jpgs)
    seeta_face_cmd = f'/data2/zjm/tools/seetface/seetaface_detection \
        /data2/zjm/tools/seetface/seeta_fd_frontal_v1.0.bin \
            {source_lst} {target_lst} 0 635454 > /dev/null 2>&1'

    os.system(seeta_face_cmd)
    print(f'Finished extract face {os.path.basename(frame_dir)}')
    time.sleep(0.2)

def get_face_openface(frame_dir):
    openface_root="/root/tools/openface_tool/OpenFace/build/bin/"
    face_dir = frame_dir.replace('Frame', 'OpenFace')
    if not os.path.exists(face_dir):
        os.mkdir(face_dir)
    
    cmd = f"{openface_root}/FaceLandmarkVidMulti -nomask -fdir {frame_dir} -out_dir {face_dir}/ > /dev/null 2>&1"
    os.system(cmd)
    print(f'Finished extract face {os.path.basename(frame_dir)}')
    time.sleep(0.2)

def pipeline(video_name):
    try:
        video_basename = os.path.basename(video_name).split('.')[0]
        frame_dir = os.path.join(output_root, video_basename)
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)
        # extract_one_video(video_name, frame_dir)
        # get_face(frame_dir)
        get_face_openface(frame_dir)

    except BaseException as e:
        log = open(os.path.join('log', 'get_face', os.path.basename(video_name).split('.')[0] + '.log'), 'w')
        log.write(traceback.format_exc())
        raise RuntimeError(traceback.format_exc())

if __name__ == "__main__":
    # set environment
    ld_seetaface = "/data2/zjm/tools/seetface"
    if not os.environ.get('LD_LIBRARY_PATH'):
        os.environ['LD_LIBRARY_PATH'] = f'{ld_seetaface}:{ld_opencv}:'
    elif ld_seetaface not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] += ld_seetaface + ':'

    raw_video_dir = '../Video'
    # output_root = '../Frame'
    output_root = "/data7/emobert/exp/evaluation/MSP-IMPROV/Frame"
    video_names = glob.glob(os.path.join(raw_video_dir, '*/*/S/*.avi'))
    # print(video_names)
    with multiprocessing.Pool(8) as pool:
        pool.map(pipeline, video_names)
    pool.join()
    pool.close()
    # pipeline("../Video/session2/S13N/S/MSP-IMPROV-S13N-F02-S-MF02.avi")
    
   
