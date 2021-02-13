import sys
import os
import os.path as osp
import multiprocessing as mp
from tqdm import tqdm
sys.path.append('/data7/MEmoBert')
from preprocess.tasks.audio import AudioSplitorTool
from preprocess.tasks.vision import VideoFaceTrackerTool, Video2FrameTool, ActiveSpeakerSelector
from preprocess.MELD.utils import find_video_using_uttid, get_config, get_int2name_label, make_or_exists


def make_frames(config=None):
    framer = Video2FrameTool()
    if config is None:
        config = get_config()
    for set_name in ['train', 'val', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        save_root = osp.join(config['output_dir'], 'frames')
        for utt_id in tqdm(int2name):
            video_path = find_video_using_uttid(config, set_name, utt_id)
            save_dir = osp.join(save_root, set_name, utt_id)
            framer(video_path, save_dir)

def make_openface(config=None, pool=None):
    facer = VideoFaceTrackerTool(openface_dir='/data2/zjm/tools/OpenFace/build/bin')
    if config is None:
        config = get_config()
    for set_name in ['train', 'val', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        save_root = osp.join(config['output_dir'], 'faces')
        make_or_exists(save_root)
        frame_root = osp.join(config['output_dir'], 'frames')
        if pool is None:
            for utt_id in tqdm(int2name):
                frame_dir = osp.join(frame_root, set_name, utt_id)
                save_dir = osp.join(save_root, set_name, utt_id)
                facer(frame_dir, save_dir)
        else:
            frame_dirs = [osp.join(frame_root, set_name, utt_id) for utt_id in int2name]
            save_dirs = [osp.join(save_root, set_name, utt_id) for utt_id in int2name] 
            print(len(frame_dirs), len(save_dirs))
            pool.starmap(facer, zip(frame_dirs, save_dirs))

def make_audio(config=None):
    spliter = AudioSplitorTool()
    if config is None:
        config = get_config()
    for set_name in ['train', 'val', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        save_root = osp.join(config['output_dir'], 'audio', set_name)
        make_or_exists(save_root)
        for utt_id in tqdm(int2name):
            video_path = find_video_using_uttid(config, set_name, utt_id)
            save_path = osp.join(save_root, utt_id + '.wav')
            spliter(video_path, save_path)

def make_active_spk(config=None):
    act = ActiveSpeakerSelector()
    if config is None:
        config = get_config()
    for set_name in ['train', 'val', 'test']:
        audio_root = osp.join(config['output_dir'], 'audio', set_name)
        face_root = osp.join(config['output_dir'], 'faces', set_name)
        int2name, _ = get_int2name_label(config, set_name)
        for utt_id in tqdm(int2name):
            face_dir = osp.join(face_root, utt_id)
            audio_path = osp.join(audio_root, utt_id + '.wav')
            act(face_dir, audio_path)

if __name__ == '__main__':
    # make_frames()

    # config = get_config()
    # path = find_video_using_uttid(config, 'val', 'dia0_utt0')
    # print(path)
    # pool = mp.Pool(24)
    # make_openface(pool=pool)

    # make_audio()
    make_active_spk()