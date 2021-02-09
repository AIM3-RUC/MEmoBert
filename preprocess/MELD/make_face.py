import sys
import os
import os.path as osp
from tqdm import tqdm
sys.path.append('/data7/MEmoBert')
from preprocess.tasks.vision import VideoFaceTrackerTool, Video2FrameTool
from preprocess.MELD.utils import find_video_using_uttid, get_config, get_int2name_label


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

def make_openface(config=None):
    facer = VideoFaceTrackerTool(openface_dir='/data2/zjm/tools/OpenFace/build/bin')
    if config is None:
        config = get_config()
    for set_name in ['train', 'val', 'test']:
        int2name, _ = get_int2name_label(config, set_name)
        save_root = osp.join(config['output_dir'], 'faces')
        for utt_id in tqdm(int2name):
            frame_dir = osp.join(config['output_dir'], 'frames', utt_id)
            save_dir = osp.join(save_root, set_name, utt_id)
            facer(frame_dir, save_dir)


if __name__ == '__main__':
    # make_frames()

    # config = get_config()
    # path = find_video_using_uttid(config, 'train', 'dia0_utt0')
    # print(path)

    make_openface()