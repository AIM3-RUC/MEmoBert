"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for extracting features
"""
from collections import defaultdict
from tqdm import tqdm

import torch
from horovod import torch as hvd
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from code.uniter.model.model import UniterModel, UniterPreTrainedModel
from code.uniter.utils.misc import NoOp

class UniterForExtracting(UniterPreTrainedModel):
    """ UNITER Extracting Features """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.apply(self.init_weights)

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
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        txt_lens = batch['txt_lens']
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        txt_sequence_output = sequence_output[:, :input_ids.size(1), :]
        # get only the img part
        img_lens = batch['img_lens']
        img_sequence_output = sequence_output[:, input_ids.size(1):, :]
        print('[Features] txt fts {} and img fts {}'.format(txt_sequence_output.size(), img_sequence_output.size()))
        return txt_sequence_output, txt_lens, img_sequence_output, img_lens
    

@torch.no_grad()
def inference(model, eval_loader):
    model.eval()
    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()
    for i, mini_batches in enumerate(eval_loader):
        j = 0
        for batch in mini_batches:
            scores = model(batch, compute_loss=False)
            bs = scores.size(0)
            score_matrix.data[i, j:j+bs] = scores.data.squeeze(1).half()
            j += bs
        assert j == score_matrix.size(1)
        pbar.update(1)
    model.train()
    pbar.close()
    return score_matrix