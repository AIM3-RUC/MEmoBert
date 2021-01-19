import os
import numpy as np
from bert_vec import BertExtractor
import glob


def extract_all_bert():
    extractor = BertExtractor(cuda=True, cuda_num=0)
    transcript_root = '../All_human_transcriptions'
    save_root = '../../MSP-IMPROV_feature/text/raw'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    txts = glob.glob(os.path.join(transcript_root, '*.txt'))
    txts = list(filter(lambda x: x.split('-')[4]=='S', txts))
    for txt in txts:
        sentence = open(txt).read().strip()
        uuid = os.path.basename(txt).split('.')[0]
        sentence_vector, _ = extractor.extract(sentence)
        sentence_vector = sentence_vector.squeeze().detach().cpu().numpy()
        save_path = os.path.join(save_root, uuid+'.npy')
        print(uuid, sentence_vector.shape)
        np.save(save_path, sentence_vector)

extract_all_bert()