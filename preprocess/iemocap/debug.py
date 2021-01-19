import torch

from transformers import BertTokenizer as TB_tokenizer
from transformers import BertModel as TB_model
from pytorch_pretrained_bert import BertModel as PB_model
from pytorch_pretrained_bert import BertTokenizer as PB_tokenizer

class BertExtractorFromWWW(object):
    def __init__(self, cuda=False, cuda_num=None):
        self.tokenizer = TB_tokenizer.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model = TB_model.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model.eval()

        if cuda:
            self.cuda = True
            self.cuda_num = cuda_num
            self.model = self.model.cuda(self.cuda_num)
        else:
            self.cuda = False

    def extract(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        print('In TB:')
        print(input_ids[0])
        print(list(map(lambda x: self.tokenizer._convert_id_to_token(x.item()), input_ids[0])))
        if self.cuda:
            input_ids = input_ids.cuda(self.cuda_num)

        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output


class BertExtractor(object):
    def __init__(self, gpu_id=None):
        self.device = torch.device(f'cuda:{gpu_id}') if gpu_id is not None else None
        # self.model = BertModel.from_pretrained('bert-base-uncased') 
        self.model = PB_model.from_pretrained('/data2/lrc/bert_cache/pytorch')
        self.model.to(self.device)
        self.model.eval()
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = PB_tokenizer.from_pretrained('/data2/lrc/bert_cache/pytorch')
    
    def extract_feat(self, text):
        ids = self.bert_tokenize(text)
        ids = [101] + ids + [102]
        print('In PB:')
        print(ids)
        print(self.tokenizer.convert_ids_to_tokens(ids))

        ids = torch.tensor(ids).unsqueeze(0)
        if self.device:
            ids = ids.to(self.device)
        with torch.no_grad():
            feat = self.model(ids)[0][0]
        return feat.squeeze().cpu().numpy()

    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids
    

if __name__ == '__main__':
    sentence = 'With the most perfect poise?'
    pb_model = BertExtractor(gpu_id=0)
    tb_model = BertExtractorFromWWW(cuda=True, cuda_num=1)
    print("Input sentence:")
    print(sentence)
    pb_feat = pb_model.extract_feat(sentence)
    tb_feat, _ = tb_model.extract(sentence)
    tb_feat = tb_feat[0].cpu().numpy()
    print('pb:', pb_feat.shape)
    print('tb:', tb_feat.shape)
    # print(pb_feat==tb_feat)
    print('---------------')
    print(pb_feat)
    print('---------------')
    print(tb_feat)
    print('---------------')
