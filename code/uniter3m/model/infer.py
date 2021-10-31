"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for extracting features
"""
from collections import defaultdict
from tqdm import tqdm

import torch
from horovod import torch as hvd
from code.uniter3m.model.model import UniterModel, UniterPreTrainedModel
from code.uniter3m.utils.misc import NoOp

'''
For extracting pretrained features or evaluation on the downstream tasks
'''

class UniterForExtracting(UniterPreTrainedModel):
    """ UNITER Extracting Features """
    def __init__(self, config, img_dim, speech_dim, use_visual, use_speech):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim, speech_dim, use_visual, use_speech)
        # self.apply(self.init_weights)

    def forward(self, batch):
        '''
        input_ids torch.Size([8, 18]) = batch['input_ids'] 
        position_ids torch.Size([1, 18]) = batch['position_ids']
        txt_lens:     list of [txt_len]
        img_feat torch.Size([8, 53, 2048]) = batch['img_feat']
        img_position_ids torch.Size([8, 53])  = batch['img_pos_feat']
        img_lens:     list of [num_faces]
        attention_mask torch.Size([8, 64]) = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        gather_index torch.Size([8, 64]) = batch['gather_index']
        return:
            text sequence output, real text len 
            img sequence output, real img len 
        '''
        batch = defaultdict(lambda: None, batch)
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        input_ids = batch['input_ids'] # (batch, max-len, dim)
        txt_lens = batch['txt_lens']
        txt_sequence_output = sequence_output[:, :input_ids.size(1), :]
        # get only the img part
        img_feats = batch['img_feat'] # (batch, max-len, dim)
        img_lens = batch['img_lens']
        img_sequence_output = sequence_output[:, input_ids.size(1) : input_ids.size(1)+img_feats.size(1), :]
        # get only the speech part
        speech_lens = batch['speech_lens']
        speech_sequence_output = sequence_output[:, input_ids.size(1)+img_feats.size(1):, :]
        # print('[Features] txt fts {} and img fts {} and speech fts {}'.format(txt_sequence_output.size(), img_sequence_output.size(), speech_sequence_output.size()))
        return txt_sequence_output, txt_lens, img_sequence_output, img_lens, speech_sequence_output, speech_lens

@torch.no_grad()
def extracting(model, loader):
    model.eval()
    if hvd.rank() == 0:
        pbar = tqdm(total=len(loader))
    else:
        pbar = NoOp()
    txt_real_sequence_outputs = []
    img_real_sequence_outputs = []
    speech_real_sequence_outputs = []
    targets = []
    for i, batch in enumerate(loader):
        txt_sequence_output, txt_lens, img_sequence_output, img_lens, speech_sequence_output, speech_lens = model(batch)
        assert txt_sequence_output.size(0) == len(txt_lens) == img_sequence_output.size(0) == len(img_lens) == speech_sequence_output.size(0) == len(speech_lens)
        for j in range(len(txt_lens)):
            # txt_len = cls + txt + sep
            txt_real_sequence_outputs.append(txt_sequence_output[j][1:txt_lens[j]-1].cpu().numpy())
            img_real_sequence_outputs.append(img_sequence_output[j][:img_lens[j]].cpu().numpy())
            speech_real_sequence_outputs.append(speech_sequence_output[j][:speech_lens[j]].cpu().numpy())
            targets.append(batch['targets'][j].cpu().numpy())
        pbar.update(1)
    pbar.close()
    print('[P{}] txt {} img {} speech {} target {}'.format(hvd.rank(), \
            len(txt_real_sequence_outputs), len(img_real_sequence_outputs), len(speech_real_sequence_outputs), len(targets)))
    return txt_real_sequence_outputs, img_real_sequence_outputs, speech_real_sequence_outputs, targets