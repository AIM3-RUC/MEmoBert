import os, glob
import shutil
import h5py
import numpy as np
import tools.istarmap
import multiprocessing
from tqdm import tqdm
from utils import get_basename, mkdir
from tasks.audio import *
from tasks.vision import *
from tasks.text import *
from tasks.common import *
from scheduler.gpu_scheduler import init_model_on_gpus
from scheduler.multiprocess_scheduler import simple_processer


def vision_process(paths):
    frame_dir = './data/frames'
    face_dir = './data/faces'
    get_frames = Video2Frame(save_root=frame_dir)
    get_faces = VideoFaceTracker(save_root=face_dir)
    get_activate_spk = ActiveSpeakerSelector()
    select_faces = FaceSelector()
    video_clip, audio_path = paths
    _frame_dir = get_frames(video_clip)
    _face_dir = get_faces(_frame_dir)
    face_paths = []
    active_spk_id = get_activate_spk(_face_dir, audio_path)
    if active_spk_id is None:
        # count['no_active_spk'] += get_basename(_frame_dir)
        pass
    else:
        face_paths = select_faces(face_dir, active_spk_id)
        if len(face_paths) < 0.4 * len(glob.glob(os.path.join(frame_dir, '*.jpg'))):
            # count['face_too_less'] += get_basename(_face_dir)
            pass
    
    return face_paths

# package_transcript = TranscriptPackager()
# ass_path = 'data/transcripts/No0001.The.Shawshank.Redemption.ass'
# transcript_info = package_transcript(ass_path)
# sentences = list(map(lambda  x: x['content'], transcript_info))
# # 检查是否句子长度>1
# _have_sentence = []
# for i, transcript in enumerate(transcript_info):
#     if len(transcript['content']) > 0:
#         _have_sentence.append(transcript)
#     else:
#         count['no_sentence'].append(transcripts['index'])

# transcript_info = _have_sentence
compile_features = CompileFeatures()
# # 过滤太短的句子
# print('[Main]: Start filtering Transcipts')
# pool = multiprocessing.Pool(8)
# filter_fun = FilterTranscrips(1)
# is_long_enough = list(tqdm(pool.imap(filter_fun, transcript_info), total=len(transcript_info)))
# transcript_info = list(filter(lambda x: x[1], zip(transcript_info, is_long_enough)))
# transcript_info = list(map(lambda x: x[0], transcript_info))
# print('[Main]: Remaining Transcipts: {}'.format(len(transcript_info)))
# bert_model = BertExtractor(device=0)
# text_contents = list(map(lambda x: x['content'], transcript_info))[:10]
# bert_features = list(tqdm(map(lambda x: bert_model(x), text_contents), total=len(text_contents)))
# for x in bert_features:
#     print(x.shape)

# bert_features, bert_lengths = compile_features(bert_features)
# print('[Main]: Bert feature:{} with length '.format(bert_features.shape, bert_lengths))
device = 0
pool = multiprocessing.Pool(8)
# audio_paths = glob.glob('data/audio_clips/No0001.The.Shawshank.Redemption/*.wav')[:10]
# print('[Main]: Total wav found:', len(audio_paths))
# print('[Main]: Extract comparE features:')
# extract_comparE = ComParEExtractor()
# comparE_features = list(tqdm(pool.imap(extract_comparE, audio_paths), total=len(audio_paths)))
# comparE_features, comparE_lengths = compile_features(comparE_features)
# print('[Main]: ComparE feature:{} with length []'.format(comparE_features.shape, comparE_lengths))
# # save_h5(comparE_features, comparE_lengths, os.path.join(movie_feature_dir, 'audio_comparE.h5'))

# print('[Main]: Extract vggish features:')
# extract_vggish = VggishExtractor(device=device)
# vggish_features = list(tqdm(map(lambda x: extract_vggish(x), audio_paths), total=len(audio_paths)))
# vggish_features, vggish_lengths = compile_features(vggish_features)
# print('[Main]: vggish feature:{} with length {}'.format(vggish_features.shape, vggish_lengths))

extract_denseface = DensefaceExtractor(device=device)
print('[Main]: start vision process:')
for i in range(100):
    clip_path = f"data/video_clips/No0001.The.Shawshank.Redemption/{i}.mkv"
    audio_path = f"data/audio_clips/No0001.The.Shawshank.Redemption/{i}.wav"
    face_paths = vision_process((clip_path, audio_path))
    if face_paths:
        print(i)
        break

# 抽denseface
print('[Main]: Extract Denseface features:')
denseface_features = []

clip_denseface_feature = list(map(lambda x: extract_denseface(x), face_paths))
clip_denseface_feature = np.asarray(clip_denseface_feature)
denseface_features.append(clip_denseface_feature)

denseface_features, denseface_lengths = compile_features(denseface_features)
print('[Main]: Denseface feature:{} with length {}'.format(denseface_features.shape, denseface_lengths))

