import os, glob
import h5py
import numpy as np
import json
from tqdm import tqdm
from utils import get_basename, mkdir
from tasks.audio import *
# from tasks.vision import *
from tasks.text import *
from tools.get_emo_words import EmoLexicon

def extract_features_h5(extract_func, get_input_func, utt_ids, save_path, multi_processing=False):
    if os.path.exists(save_path):
        try:
            _h5f = h5py.File(save_path, 'r')
            if len(_h5f.keys()) == len(utt_ids):
                _h5f.close()
                return
            _h5f.close()
        except OSError:
            os.remove(save_path)

    h5f = h5py.File(save_path, 'w')

    for utt_id in tqdm(utt_ids):
        input_param = get_input_func(utt_id)
        feature = extract_func(input_param)
        h5f[utt_id] = feature
    h5f.close()

def get_utt_id_files(meta_dir, file_name):
    files = glob.glob(os.path.join(meta_dir, f'*/{file_name}.txt'))
    files = sorted(files)
    movie_names = list(map(lambda x: x.split('/')[-2], files))
    return files, movie_names

def process_emo_word(utt_ids, emol, save_path, multiprocessing=False):
    if os.path.exists(save_path):
        content = json.load(open(save_path))
        if len(content.keys()) == len(utt_ids):
            return 
    
    json_path = utt_ids[0].split('/')[0]
    json_path = os.path.join(transcript_dir, json_path + '.json')
    utts = json.load(open(json_path))
    if not multiprocessing:
        all_word2affect = {}
        for utt_id in tqdm(utt_ids):
            utterance = utts[utt_id.split('/')[-1]]['content']
            emo_words, word2affect = emol.get_emo_words(utterance)
            all_word2affect[utt_id] = word2affect
    else:
        pool = multiprocessing.Pool(4)
        utterances = list(map(lambda x: utts[x.split('/')[-1]]['content'], utt_ids))
        ret = pool.map(emol.get_emo_words, utt_ids)
        all_word2affect = list(map(lambda x:x[1], ret))
        all_word2affect = dict([(utt_id, word2affect) for utt_id, word2affect in zip(utt_ids, all_word2affect)])

    json.dump(all_word2affect, open(save_path, 'w'), indent=4)


if __name__ == '__main__':
    import sys
    utt_file_name = sys.argv[1]
    part_no, total = eval(sys.argv[2]), eval(sys.argv[3])
    device = 0
    meta_dir = 'data/meta'
    feature_root = 'feature'
    face_dir = 'data/faces'
    frame_dir = 'data/frames'
    audio_dir = 'data/audio_clips'
    transcript_dir = 'data/transcripts/json'
    tmp_dir = 'data/.tmp'
    extract_comparE = ComParEExtractor(tmp_dir=tmp_dir)
    extract_vggish = VggishExtractor(device=device)
    lexicon_dir = '/data2/zjm/tools/EmoLexicons'
    lexicon_name = 'LIWC2015Dictionary.dic'
    utterance = "Because you're going to get us all fucking pinched. What are you, so stupid?".lower()
    bert_vocab_filepath = '/data2/zjm/tools/LMs/bert_base_en/vocab.txt'
    emol = EmoLexicon(lexicon_dir, lexicon_name, is_bert_token=True, bert_vocab_filepath=bert_vocab_filepath)
        
    # extract_denseface = DensefaceExtractor(device=device)
    all_utt_files, movie_names = get_utt_id_files(meta_dir, utt_file_name)
    length = len(all_utt_files)
    start = int(part_no * length / total)
    end = int((part_no + 1) * length / total)
    
    print('[Main]: all utt_id files found:', len(all_utt_files))
    print('-------------------------------------------------')
    for i, movie_name in enumerate(movie_names):
        print(f'[{i}]\t{movie_name}')
    print('-------------------------------------------------')
    print('[Main]: movies to be processed:')
    print('-------------------------------------------------')
    all_utt_files = all_utt_files[start: end]
    movie_names = movie_names[start: end]
    for i, movie_name in zip(range(start, end), movie_names):
        print(f'[{i}]\t{movie_name}')
    print('-------------------------------------------------')
    for utt_file, movie_name in zip(all_utt_files, movie_names):
        print(f'[Main]: Process {movie_name}:')
        feature_dir = os.path.join(feature_root, movie_name)
        mkdir(feature_dir)
        utt_ids = open(utt_file).readlines()
        utt_ids = list(map(lambda x: x.strip(), utt_ids))
        # comparE
        print('[Main]: processing comparE')
        save_path = os.path.join(feature_dir, f'{utt_file_name}_comparE.h5')
        extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x+'.wav'), 
                    utt_ids, save_path)
        print(f'[ComparE]: {movie_name} saved in {save_path}')
        # vggish
        print('[Main]: processing vggish')
        save_path = os.path.join(feature_dir, f'{utt_file_name}_vggish.h5')
        extract_features_h5(extract_vggish, lambda x: os.path.join(audio_dir, x+'.wav'), 
                    utt_ids, save_path)
        print(f'[Vggish]: {movie_name} saved in {save_path}')
        # # denseface
        # save_path = os.path.join(feature_dir, f'{utt_file_name}_vggish.h5')
        # extract_features_h5(extract_denseface, lambda x: os.path.join(audio_dir, x+'.wav'), 
        #             utt_ids, save_path)

        # emo_word
        save_path = os.path.join(feature_dir, f'{utt_file_name}_emoword.json')
        process_emo_word(utt_ids, emol, save_path, multiprocessing=False)
        print(f'[EmoWord]: {movie_name} saved in {save_path}')
        


    
