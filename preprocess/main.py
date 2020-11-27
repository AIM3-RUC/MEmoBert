import os, glob
import h5py
import numpy as np
import multiprocessing
from tqdm import tqdm
from utils import get_basename, mkdir
from tasks.audio import *
from tasks.vision import *
from tasks.text import *
from tasks.common import *

def save_h5(feature, lengths, save_path):
    h5f = h5py.File(save_path, 'w')
    h5f['feature'] = feature
    h5f['lengths'] = lengths
    h5f.close()

if __name__ == '__main__':
    import sys

    audio = True if 'audio' in sys.argv[1:] else False
    vision = True if 'vision' in sys.argv[1:] else False
    text = True if 'text' in sys.argv[1:] else False
    print(f'Modality: Audio {audio}, Vision:{vision}, Text:{text}')

    pool = multiprocessing.Pool(8)

    transcripts_dir = './data/transcripts'
    video_clip_dir = './data/video_clips'
    audio_dir = './data/audio_clips'
    frame_dir = './data/frames'
    face_dir = './data/faces'
    comparE_tmp = './data/.tmp'
    feature_dir = './feature'

    # 流程
    extract_text = TranscriptExtractor(save_root=transcripts_dir)
    package_transcript = TranscriptPackager()
    cut_video = VideoCutterOneClip(save_root=video_clip_dir) # VideoCutter(save_root=video_clip_dir)
    
    device = 0
    # 语音
    if audio or vision:
        extract_audio = AudioSplitor(save_root=audio_dir)
    if audio:
        extract_vggish = VggishExtractor(device=device)
        extract_comparE = ComParEExtractor(tmp_dir=comparE_tmp)

    # 视觉
    if vision:
        get_frames = Video2Frame(save_root=frame_dir)
        get_faces = VideoFaceTracker(save_root=face_dir)
        get_activate_spk = ActiveSpeakerSelector()
        select_faces = FaceSelector()
        extract_denseface = DensefaceExtractor(device=device)

    # 文本
    if text:
        extract_bert = BertExtractor(device=device)

    # 特征收集
    compile_features = CompileFeatures()
    vision_feature = []
    audio_feature = []
    
    all_movies = glob.glob('/data6/zjm/emobert/resources/raw_movies/*.mkv')
    for movie in all_movies:
        print('Processing', movie)
        transcript_path = extract_text(movie)
        transcript_info = package_transcript(transcript_path)
        
        if transcript_info == None:
            continue
        # 切视频
        print('Start cutting video')
        # list(tqdm(pool.imap(cut_video(movie, transcript_info[:3])), total=3))
        video_clip_dir = pool.starmap(cut_video, [(movie, transcript, i) for i, transcript in enumerate(transcript_info)])[0]
        all_video_clip = sorted(glob.glob(os.path.join(video_clip_dir, '*.mkv')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
        print('Total clips found:', len(all_video_clip))

        all_comparE = []
        all_vggish = []
        all_denseface = []
        all_bert = []

        # 切语音:
        audio_paths = pool.map(extract_audio, all_video_clip)

        for video_clip, transcript, audio_path in list(zip(all_video_clip, transcript_info, audio_paths)):
            # 提语音
            if audio:
                comparE_feature = extract_comparE(audio_path)
                vggish_feature = extract_vggish(audio_path)
                if not (isinstance(comparE_feature, np.ndarray) and isinstance(vggish_feature, np.ndarray)):
                    continue
            # 提视觉
            if vision:
                frame_dir = get_frames(video_clip)
                input()
                face_dir = get_faces(frame_dir)
                active_spk_id = get_activate_spk(face_dir, audio_path)
                if active_spk_id is None:
                    continue
            
                face_paths = select_faces(face_dir, active_spk_id)
                face_features = []
                for face_path in face_paths:
                    denseface_feature = extract_denseface(face_path)
                    face_features.append(face_features)
                face_features = np.concatenate(face_features)
                
            # 提文本
            if text:
                bert_feature = extract_bert(transcript['content'])
                
            # 保存特征
            if audio:
                all_comparE.append(comparE_feature)
                all_vggish.append(vggish_feature)
            if vision:
                all_denseface.append(face_features)
            if text:
                all_bert.append(bert_feature.squeeze(0))
        
        basename = get_basename(movie)
        save_dir = os.path.join(feature_dir, basename)
        mkdir(save_dir)
        # 保存特征
        if audio:
            print("ComparE:")
            comparE_features, comparE_lengths = compile_features(all_comparE)
            print(comparE_lengths, comparE_features.shape)
            save_h5(comparE_features, comparE_lengths, os.path.join(save_dir, 'audio_comparE.h5'))

            print("vggish:")
            vggish_features, vggish_lengths = compile_features(all_vggish)
            print(vggish_lengths, vggish_features.shape)
            save_h5(vggish_features, vggish_lengths, os.path.join(save_dir, 'audio_vggish.h5'))

        if vision:
            print("denseface:")
            denseface_features, denseface_lengths = compile_features(all_denseface)
            print(denseface_lengths, denseface_features.shape)
            save_h5(denseface_features, denseface_lengths, os.path.join(save_dir, 'vision_denseface.h5'))
        
        if text:
            print("Bert")
            bert_features, bert_lengths = compile_features(all_bert)
            print(bert_lengths, bert_features.shape)
            save_h5(bert_features, bert_lengths, os.path.join(save_dir, 'text_bert.h5'))


        