import os
import os.path as osp
import glob
import json
from tqdm import tqdm
import multiprocessing as mp
from preprocess.tasks.common import FilterTranscrips

def extract_full_audio(movie_path, save_path):
    _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(movie_path, save_path)
    os.system(_cmd)

def extract_clips(full_audio_path, start, end, save_path):
    _cmd = f"ffmpeg -i {full_audio_path} -vn -acodec copy -ss {start} -to {end} {save_path} -y > /dev/null 2>&1"
    os.system(_cmd)

def find_movie_path_by_name(movie_name):
    root1 = '/data7/MEmoBert/emobert/resources/raw_movies'
    root2 = '/data7/MEmoBert/emobert/resources/raw_movies_v1'
    for root in [root1, root2]:
        for _format in ['mp4', 'mkv']:
            path = osp.join(root, movie_name + '.' + _format)
            if osp.exists(path):
                return path
    return None

def find_all_transcripts():
    transcript_dir = '/data7/MEmoBert/emobert/data_nomask_new/transcripts/json'
    return glob.glob(osp.join(transcript_dir, '*.json'))

def filter_transcripts(transcript_info):
    ## 过滤掉没有句子的
    _have_sentence = {}
    for key, info in transcript_info.items():
        if len(info['content']) > 0:
            _have_sentence[key] = info
    transcript_info = _have_sentence
    # 过滤掉时间太短的
    filter_transcript = FilterTranscrips(threshold=1.5)
    transcript_info = list(filter(lambda x: filter_transcript(x[1]), transcript_info.items()))
    transcript_info = dict(transcript_info)
    return transcript_info

def process_one_movie(movie_path, transcript_path):
    transcript_data = json.load(open(transcript_path))
    transcript_data = filter_transcripts(transcript_data)
    movie_name = get_movie_name(movie_path)
    full_wav_store = "/data7/lrc/movie_dataset/full_wav"
    full_wav_path = osp.join(full_wav_store, movie_name + '.wav')
    clip_wav_store = osp.join("/data7/lrc/movie_dataset/clips", movie_name)
    if not os.path.exists(clip_wav_store):
        os.makedirs(clip_wav_store)

    extract_full_audio(movie_path, full_wav_path)
    for key in transcript_data.keys():
        sentence_info = transcript_data[key]
        start = sentence_info['start']
        end = sentence_info['end']
        save_path = osp.join(clip_wav_store, str(key)+".wav")
        extract_clips(full_wav_path, start, end, save_path)

def get_movie_name(movie_path):
    basename = osp.basename(movie_path)
    return basename[:basename.rfind('.')]

def process_one_movie_pack(script_path):
    movie_name = get_movie_name(script_path)
    movie_path = find_movie_path_by_name(movie_name)
    if movie_path is None:
        return
    process_one_movie(movie_path, script_path)

def process_all_movie():
    all_scripts = find_all_transcripts()
    print(f'All script founded :{len(all_scripts)}')
    print('--------------------------------------------')
    for i, script in enumerate(all_scripts):
        print(i, get_movie_name(script))
    print('--------------------------------------------')

    # for script in tqdm(all_scripts):
    #     movie_name = get_movie_name(script)
    #     movie_path = find_movie_path_by_name(movie_name)
    #     if not movie_name:
    #         continue
    #     process_one_movie(movie_path, script)
    pool = mp.Pool(20)
    list(tqdm(pool.imap(process_one_movie_pack, all_scripts), total=len(all_scripts)))
    
if __name__ == '__main__':
    process_all_movie()
    #
    