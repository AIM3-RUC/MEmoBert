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

from code.uniter3m.model.model import UniterModel, UniterPreTrainedModel
from code.uniter3m.model.lare_layers import BertPostagHead, BertSentiHead, BertPolarityHead
## from uniter 
from code.uniter.model.pretrain import RegionFeatureRegression, RegionClassification
from code.uniter.model.layer import GELU, BertOnlyMLMHead, BertPredictionHeadTransform

class EmoClassification(nn.Module):
    " for the emotion classification, with kl-loss"
    def __init__(self, hidden_size, label_dim, cls_dropout=0.1, cls_type='small_vqa'):
        super().__init__()
        if cls_type == 'emocls':
            self.output = nn.Sequential(
                nn.Dropout(cls_dropout),
                nn.Linear(hidden_size, hidden_size),
                GELU(),
                nn.Dropout(cls_dropout),
                nn.Linear(hidden_size, label_dim)
                )
        elif cls_type == 'vqa':
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size*2), 
                GELU(),
                LayerNorm(hidden_size*2, eps=1e-12),
                nn.Linear(hidden_size*2, label_dim)
                )
        elif cls_type == 'small_vqa':
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                GELU(),
                LayerNorm(hidden_size, eps=1e-12),
                nn.Linear(hidden_size, label_dim)
                )
        else:
            print("------- [Error] classifier type {}".format(cls_type))
            exit(0)

    def forward(self, input_):
        output = self.output(input_)
        return output

class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """
    def __init__(self, config, img_dim, speech_dim, img_label_dim, use_visual, use_speech):
        super().__init__(config)
        self.config = config
        self.use_speech = use_speech
        self.use_visual = use_visual
        self.uniter = UniterModel(config, img_dim, speech_dim, use_visual=self.use_visual, 
                                    use_speech=self.use_speech)

        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)

        if self.use_visual:
            print('[Debug] use visual feature regression and region classification!!!')
            self.feat_regress = RegionFeatureRegression(
                config.hidden_size, img_dim,
                self.uniter.img_embeddings.img_linear.weight)
            self.region_classifier = RegionClassification(
                config.hidden_size, img_label_dim)
        
        if self.use_speech:
            print('[Debug] use speech feature regression!!!')
            self.speech_feat_regress = RegionFeatureRegression(
                config.hidden_size, speech_dim,
                self.uniter.speech_embeddings.speech_linear.weight)

        # Jinming: add for melm multi-task
        if config.melm_multitask is True:
            print("Use the melm multitask, default use the ")
            self.emomelm_classifier = EmoClassification(
                config.hidden_size, config.melm_emo_category_size, cls_type='vqa')

        if config.use_emolare:
            print("Use the emolare module")
            self.pos_tag_predictions = BertPostagHead(config, self.uniter.embeddings.pos_tag_embedding.weight)
            self.senti_word_predictions = BertSentiHead(config, self.uniter.embeddings.word_senti_embedding.weight)
            self.senti_utt_predictions = BertPolarityHead(config, self.uniter.embeddings.utt_senti_embedding.weight)

        # for emotion classification
        self.emo_classifier = EmoClassification(
                config.hidden_size, config.weak_emo_category_size, cls_type='vqa')
        # for image-text matching
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.eitm_output = nn.Linear(config.hidden_size, 2)
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
            return self.forward_mlm(batch, txt_labels, compute_loss=compute_loss)
        elif task == 'melm':
            txt_labels = batch['txt_labels']
            # jinming: add emo labels is None or int
            if self.config.melm_multitask:
                txt_emo_labels = batch['txt_emo_labels']
            else:
                txt_emo_labels = None
            # print('[Debug in MELM forward] the txt_emo_labels {}'.format(txt_emo_labels))
            return self.forward_melm(batch, txt_labels, txt_emo_labels, compute_loss=compute_loss)
        elif task == 'mrfr' or task == 'merfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(batch, img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss=compute_loss)
        elif task == 'msrfr':
            speech_mask_tgt = batch['speech_mask_tgt']
            speech_masks = batch['speech_masks']
            msrfr_feat_target = batch['feat_targets']
            return self.forward_msrfr(batch, speech_masks, speech_mask_tgt,
                                     msrfr_feat_target, compute_loss=compute_loss)
        elif task == 'itm' or task == 'vtm' or task == 'stm' or task == 'eitm':
            targets = batch['targets']
            return self.forward_itm(batch, targets, compute_loss=compute_loss)
        elif task == 'eitm':
            targets = batch['targets']
            return self.forward_eitm(batch, targets, compute_loss=compute_loss)
        elif task.startswith('mrc') or task.startswith('merc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(batch, img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss=compute_loss)
        elif task == 'emocls':
            targets = batch['targets']
            return self.forward_emocls(batch, targets, compute_loss=compute_loss)
        elif task == 'emolare':
            txt_labels = batch['txt_labels']
            return self.forward_emolare(batch, txt_labels, compute_loss=compute_loss)
        else:
            raise ValueError(f'invalid task {task}')

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
            # print('[Debug] in MLM and lmloss {}'.format(masked_lm_loss.shape)) # all validate token loss torch.Size([35])
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
            if self.config.melm_multitask and txt_emo_labels is not None:
                prediction_emo_scores = self.emomelm_classifier(masked_output)
                masked_emo_loss = F.cross_entropy(prediction_emo_scores, 
                                                    txt_emo_labels[txt_emo_labels != -1],
                                                    reduction='none')
                # print('[Debug] in MELM emoloss {}'.format(masked_emo_loss))
                # print('[Debug] in MELM lmloss {}'.format(masked_lm_loss))
                masked_lm_loss += self.config.melm_multitask_rate * masked_emo_loss
            # print('[Debug] in MLM lmloss {}'.format(masked_lm_loss))
            return masked_lm_loss
        else:
            # jinming: add multitask emo classification
            if self.config.melm_multitask and txt_emo_labels is not None:
                prediction_emo_scores = self.emomelm_classifier(masked_output)
                return (prediction_scores, prediction_emo_scores)
            else:
                return prediction_scores
    
    def forward_emolare(self, batch, txt_labels, use_emolare_input=True, compute_loss=True):
        '''
        use_emolare_input: must be true
        Early Fusion of Emo LARE. 没有句子级别的情感分类层, 
        Late Supervised 在EF的基础上加了一个句子分类层,
        目前一个batch里面既有EF又有LS, 但是格式是一样的，所以二者几乎是一样的。
        当输入的utt-category是unknown的时候类别是什么呢？ 此时的label全是-1，所以没有loss.
        four loss: mlm, postag, word_senti, utt_senti.
        注意1: 计算Loss的时候需要注意，token, pos, word-senti 由于unknow词不需要预测pos和wsenti. 另外又由于 EF 和 LS 混合在一块，所以有个samples不需要预测 utt-sentiment, 
        此时, 几个loss的batchsize都不一致, 因此需要在不需要预测位置loss进行补0.
        注意2：多个loss需要对齐，计算的时候不能根据label=-1进行mask操作.
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        sequence_output = self.uniter(batch, use_emolare_input, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        prediction_scores = self.cls(sequence_output)
        # print(f'[Debug] prediction_scores {prediction_scores.shape}') # torch.Size([20, 11, 30522])
        
        # for pos tag precition
        txt_pos_labels = batch['txt_pos_labels']
        pos_prediction_scores = self.pos_tag_predictions(sequence_output)
        # print(f'[Debug] pos_prediction_scores {pos_prediction_scores.shape}')

        # for word senti precition
        txt_senti_labels = batch['txt_senti_labels']
        wsenti_prediction_scores = self.senti_word_predictions(sequence_output)
        # print(f'[Debug] wsenti_prediction_scores {wsenti_prediction_scores.shape}')

        # for utt senti precition
        sentence_polarity_labels = batch['sentence_polarity_label']
        usenti_prediction_scores = self.senti_utt_predictions(sequence_output)
        # print(f'[Debug] sentence_polarity_labels {sentence_polarity_labels.shape}')
        # print(f'[Debug] usenti_prediction_scores {usenti_prediction_scores.shape}')

        masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                            txt_labels.view(-1), reduction='none', ignore_index=-1)
        # print(f'[Debug] masked_lm_loss {masked_lm_loss}')  ## 绝大部分都是0
        # print(f'[Debug] masked_lm_loss {masked_lm_loss.shape}')  ## torch.Size([220])
        # 先都设置初始值为0
        masked_pos_loss = torch.zeros_like(masked_lm_loss, device=masked_lm_loss.device)
        masked_wsenti_loss = torch.zeros_like(masked_lm_loss, device=masked_lm_loss.device)
        masked_usenti_loss = torch.zeros_like(masked_lm_loss, device=masked_lm_loss.device)
        # 然后分别计算
        # print(f'max-pos {torch.max(txt_pos_labels)}')
        # print(f'max-wsenti {torch.max(txt_senti_labels)}')
        # print(f'max-usenti {torch.max(sentence_polarity_labels)}')
        masked_pos_loss += F.cross_entropy(pos_prediction_scores.view(-1, 5),
                                            txt_pos_labels.view(-1),
                                            reduction='none', ignore_index=-1)
        masked_wsenti_loss += F.cross_entropy(wsenti_prediction_scores.view(-1, 3),
                                            txt_senti_labels.view(-1),
                                            reduction='none', ignore_index=-1)
        masked_usenti_loss += F.cross_entropy(usenti_prediction_scores.view(-1, 6),
                                            sentence_polarity_labels.view(-1),
                                            reduction='none', ignore_index=-1)
        # print(f'[Debug] masked_pos_loss {masked_pos_loss.shape}')
        # print(f'[Debug] masked_wsenti_loss {masked_wsenti_loss.shape}')
        # print(f'[Debug] masked_usenti_loss {masked_usenti_loss.shape}')
        if compute_loss:
            total_loss = masked_lm_loss + masked_pos_loss + masked_wsenti_loss + masked_usenti_loss
            # print('[Debug] total emolare loss {}'.format(total_loss.shape))
            return total_loss
        else:
            return prediction_scores, pos_prediction_scores, wsenti_prediction_scores, usenti_prediction_scores, \
                            masked_lm_loss, masked_pos_loss, masked_wsenti_loss, masked_usenti_loss

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
    
    def forward_msrfr(self, batch, speech_masks, speech_mask_tgt,
                     feat_targets, compute_loss=True):

        sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                      speech_masks=speech_masks)
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    speech_mask_tgt)
        prediction_feat = self.speech_feat_regress(masked_output)

        if compute_loss:
            msrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return msrfr_loss
        else:
            return prediction_feat


    def forward_itm(self, batch, targets, compute_loss=True):
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss
        else:
            return itm_scores
    
    def forward_eitm(self, batch, targets, compute_loss=True):
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        eitm_scores = self.eitm_output(pooled_output)
        if compute_loss:
            eitm_loss = F.cross_entropy(eitm_scores, targets, reduction='none')
            return eitm_loss
        else:
            return eitm_scores
    
    def forward_emocls(self, batch, targets, compute_loss=True):
        '''
        targets: probs or logits or hard-category
            emocls_type: soft using kl-loss, logits using kl-loss, hard using ce-loss
        '''
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        prediction_soft_label = self.emo_classifier(pooled_output) # logits

        if compute_loss:
            if  self.config.emocls_type == 'soft':
                # print('[Debug] using the soft emo cls method')
                prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
                # the target should be the softmax(logits), donot do the log
                emocls_loss = F.kl_div(prediction_soft_label, targets, reduction='none', log_target=False)
                return emocls_loss
            elif self.config.emocls_type == 'logits':
                # get temperture probs
                # print('[Debug] using the logits/temp method')
                targets = targets.true_divide(self.config.emocls_temperture)
                prediction_soft_label = prediction_soft_label.true_divide(self.config.emocls_temperture)
                prediction_soft_label = F.log_softmax(prediction_soft_label, dim=-1)
                # the target should be the softmax(logits), donot do the log
                emocls_loss = F.kl_div(prediction_soft_label, targets, reduction='none', log_target=False)
                return emocls_loss
            elif self.config.emocls_type == 'hard':
                # print('[Debug] using the hard emo cls method')
                emocls_loss = F.cross_entropy(prediction_soft_label, targets, reduction='none')
                return emocls_loss
            else:
                print('[Error] the emocls_type {}'.format(self.config.emocls_type))
        else:
            return prediction_soft_label

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
                # the target should be the softmax(logits), donot do the log
                mrc_loss = F.kl_div(prediction_soft_label, label_targets, reduction='none', log_target=False)
            else:
                print('[Error] of the loss type')
                exit(0)
            return mrc_loss
        else:
            return prediction_soft_label