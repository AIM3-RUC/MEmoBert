"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict
import logging

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from code.uniter3flow.model.layer import GELU, BertOnlyMLMHead
from code.uniter3flow.model.model_base import BertConfig
from code.uniter3flow.model.model import MEmoBertModel

logger = logging.getLogger(__name__)

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

class MEmoBertForPretraining(nn.Module):
    """ MEmoBert pretraining 
    classifier的初始化部分采用 cross-encoder 部分的参数, self.emoBert.c_config
    """
    def __init__(self, config_file, use_speech, use_visual, 
                                            pretrained_text_checkpoint=None, 
                                            pretrained_audio_checkpoint=None,
                                            fix_text_encoder=False,
                                            fix_visual_encoder=False,
                                            fix_speech_encoder=False,
                                            fix_cross_encoder=False,
                                            use_type_embedding=False):
        super(MEmoBertForPretraining, self).__init__()
        config = BertConfig.from_json_file(config_file)
        # logger.info('[Debug] Config {}'.format(type(config))) # BertConfig
        self.emoBert = MEmoBertModel(config, use_speech, use_visual, 
                                            pretrained_text_checkpoint=pretrained_text_checkpoint,
                                            pretrained_audio_checkpoint=pretrained_audio_checkpoint,
                                            fix_text_encoder=fix_text_encoder,
                                            fix_visual_encoder=fix_visual_encoder,
                                            fix_speech_encoder=fix_speech_encoder,
                                            fix_cross_encoder=fix_cross_encoder,
                                            use_type_embedding=use_type_embedding)
        logger.info('[Debug] MEmoBertModel Success!!!')
        self.cls = BertOnlyMLMHead(
            self.emoBert.c_config, self.emoBert.text_encoder.embeddings.word_embeddings.weight)
        # Jinming: add for melm multi-task
        if self.emoBert.c_config.melm_multitask is True:
            logger.info("Use the melm multitask")
            self.emomelm_classifier = EmoMelmClassification(
            self.emoBert.c_config.hidden_size, self.emoBert.c_config.melm_type_emo_size)
        self.itm_output = nn.Linear(self.emoBert.c_config.hidden_size, 2)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.emoBert.c_config.initializer_range)
        elif isinstance(module, (LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight.data)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch, task, compute_loss=True):
        '''
        input_ids torch.Size([8, 18]) = batch['input_ids'] 
        position_ids torch.Size([1, 18]) = batch['position_ids']
        img_feat torch.Size([8, 53, 2048]) = batch['img_feat']
        img_position_ids torch.Size([8, 53])  = batch['img_pos_feat']
        attention_mask torch.Size([8, 64]) = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
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
        elif task == 'itm':
            targets = batch['targets']
            return self.forward_itm(batch, targets, compute_loss)
        elif task.startswith('fom'):
            # frame order modeling
            pass
        elif task.startswith('som'):
            # speech order modeling
            pass
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, batch, txt_labels, compute_loss=True):
        '''
        利用encoder最后一层的输出进行预测, 
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        sequence_output = self.emoBert(batch, output_all_encoded_layers=False)
        # get only the text part
        # print('[Debug in MLM] input_ids {}'.format(input_ids.shape))
        # print('[Debug in MLM] sequence_output {}'.format(sequence_output.shape))
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
        sequence_output = self.emoBert(batch, output_all_encoded_layers=False)
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

    def forward_itm(self, batch, targets, compute_loss=True):
        sequence_output = self.emoBert(batch, output_all_encoded_layers=False)
        pooled_output = self.emoBert.cross_encoder.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)
        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss
        else:
            return itm_scores