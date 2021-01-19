import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class BertExtractor(object):
    def __init__(self, cuda=False, cuda_num=None):
        self.tokenizer = BertTokenizer.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model = BertModel.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

if __name__ == '__main__':
    import numpy as np
    a = BertExtractor()
    text = "Here is the sentence I want embeddings for."
    # marked_text = "[CLS] " + text + " [SEP]"
    marked_text = "[CLS]"
    # word_embedding, sentence_embedding = a.extract(['I', 'am', 'a', 'tool', 'man', '∂ß®©˙ƒ∆∫√ç'])
    word_embedding, sentence_embedding = a.extract(marked_text)
    print(word_embedding.shape)
    print(sentence_embedding.shape)
    print(word_embedding)
    word_embedding = word_embedding.squeeze()
    word_embedding = word_embedding.cpu().numpy()
    np.save('/data2/ljj/sser_discrete_data/iemocap/feature/text/start_embd.npy', word_embedding)

