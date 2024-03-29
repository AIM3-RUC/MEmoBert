"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from code.uniter.model.layer import GELU, BertOnlyMLMHead
from code.uniter.model.model import UniterModel, UniterPreTrainedModel

class RegionFeatureRegression(nn.Module):
    " for MRM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class EmoMelmClassification(nn.Module):
    " for the multitask of MELM"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        # print('[Debug] in EmoMelmClassification input {}'.format(input_.shape))
        # print('[Debug] in EmoMelmClassification output {}'.format(output.shape))
        return output

class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.config = config
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        
        # Jinming: add for melm multi-task
        if config.melm_multitask is True:
            print("Use the melm multitask")
            self.emomelm_classifier = EmoMelmClassification(
                config.hidden_size, config.melm_emo_category_size
            )
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        '''
        input_ids torch.Size([8, 18]) = batch['input_ids'] 
        position_ids torch.Size([1, 18]) = batch['position_ids']
        img_feat torch.Size([8, 53, 2048]) = batch['img_feat']
        img_position_ids torch.Size([8, 53])  = batch['img_pos_feat']
        attention_mask torch.Size([8, 64]) = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        gather_index torch.Size([8, 64]) = batch['gather_index']
        '''
        batch = defaultdict(lambda: None, batch)
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(batch, txt_labels, compute_loss)
        elif task == 'melm':
            txt_labels = batch['txt_labels']
            # jinming: add emo labels is None or int
            if self.config.melm_multitask:
                txt_emo_labels = batch['txt_emo_labels']
            else:
                txt_emo_labels = None
            return self.forward_melm(batch, txt_labels, txt_emo_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(batch, img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(batch, targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(batch, img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task.startswith('caption'):
            # caption task
            txt_labels = batch['txt_labels']
            return self.forward_caption(batch, txt_labels, compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, batch, txt_labels, compute_loss=True):
        '''
        利用encoder最后一层的输出进行预测, 
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores
    
    def forward_melm(self, batch, txt_labels, txt_emo_labels=None, compute_loss=True):
        '''
        利用encoder最后一层的输出进行预测, + 对预测的词进行情感分类
        txt_emo_labels: if none, then donot use multi-task else use multi-task
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            # jinming: add multitask emo classification
            if txt_emo_labels is not None:
                prediction_emo_scores = self.emomelm_classifier(masked_output)
                masked_emo_loss = F.cross_entropy(prediction_emo_scores, 
                                                    txt_emo_labels[txt_emo_labels != -1],
                                                    reduction='none')
                # print('[Debug] in MELM emoloss {}'.format(masked_emo_loss))
                # print('[Debug] in MELM lmloss {}'.format(masked_lm_loss))
                # 两个loss处于相同的量级，所以设置 melm_multitask_rate=1.0
                masked_lm_loss += self.config.melm_multitask_rate * masked_emo_loss
            return masked_lm_loss
        else:
            # jinming: add multitask emo classification
            if txt_emo_labels is not None:
                prediction_emo_scores = self.emomelm_classifier(masked_output)
                return (prediction_scores, prediction_emo_scores)
            else:
                return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, batch, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):

        sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                      img_masks=img_masks)
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, batch, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

    def forward_mrc(self, batch, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label