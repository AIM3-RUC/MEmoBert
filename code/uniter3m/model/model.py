"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open

import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from code.uniter.model.layer import BertLayer, BertPooler
from code.uniter.model.model import UniterImageEmbeddings

logger = logging.getLogger(__name__)

class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model

class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        # build position vocab embeddings = 512
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)      

        ## for SentiLARE module
        if self.config.use_word_senti_embedding:
            print("[Debug] use_word_senti_embedding")
            # index=4, means other category 
            self.word_senti_embedding = nn.Embedding(3, self.config.hidden_size, padding_idx=2)
        else:
            self.word_senti_embedding = None
        if self.config.use_pos_tag_embedding:
            print("[Debug] use_pos_tag_embedding")
            # index=4, means other category 
            self.pos_tag_embedding = nn.Embedding(5, self.config.hidden_size, padding_idx=4)
        else:
            self.pos_tag_embedding = None
        if self.config.use_utt_senti_embedding:
            print("[Debug] use_utt_senti_embedding")
            # index=5, means unknown category 
            self.utt_senti_embedding = nn.Embedding(6, self.config.hidden_size, padding_idx=5)
        else:
            self.utt_senti_embedding = None
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None,
                    token_pos_tag_ids=None, token_senti_ids=None,
                    token_utt_senti_ids=None):
        '''
        emo_type_ids: the emotion types of the input ids
        batch-data
        '''
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        if token_pos_tag_ids is None:
            token_pos_tag_embeddings = 0
        else:
            token_pos_tag_embeddings = self.pos_tag_embedding(token_pos_tag_ids)

        if token_senti_ids is None:
            token_senti_embeddings = 0
        else:
            token_senti_embeddings = self.word_senti_embedding(token_senti_ids)

        if token_utt_senti_ids is None:
            token_utt_senti_embeddings = 0
        else:
            token_utt_senti_embeddings = self.utt_senti_embedding(token_utt_senti_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings
                      + token_pos_tag_embeddings
                      + token_senti_embeddings
                      + token_utt_senti_embeddings)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class UniterSpeechEmbeddings(nn.Module):
    def __init__(self, config, speech_dim):
        super().__init__()
        '''
        # Jinming: 因为prtrained的bert只有两个token-type, 
        # 因此当不采用visual信息的时候,可以采用共享文本的token-type.
        '''
        self.speech_linear = nn.Linear(speech_dim, config.hidden_size)

        if config.use_projs_av_modality:
            logger.info('[Debug] add one more linear for speech modality')
            self.projs_linear = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                GELU(),
                FusedLayerNorm(config.hidden_size, eps=1e-12),
                nn.Dropout(config.hidden_dropout_prob)
            )
        else:
            self.projs_linear = None
            
        self.speech_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.mask_embedding = nn.Embedding(2, speech_dim, padding_idx=0)

        if config.speech_visual_use_same_type:
            logger.info('[Debug] use the same type embedding with visual')
        else:
            logger.info('[Debug] use the independently type embedding for speech')
            self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, speech_feat, speech_position_ids, speech_type_embeddings, speech_masks=None):
        if speech_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(speech_masks.long())
            speech_feat = speech_feat + mask

        if speech_type_embeddings is None:
            speech_type_ids = torch.zeros_like(speech_feat[:, :, 0].long())
            speech_type_embeddings = self.token_type_embeddings(speech_type_ids)

        transformed_speech = self.speech_layer_norm(self.speech_linear(speech_feat))
        if self.projs_linear is not None:
            transformed_speech = self.projs_linear(transformed_speech)

        position_embeddings = self.position_embeddings(speech_position_ids)
        embeddings = transformed_speech + position_embeddings + speech_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask, frozen_en_layers=0,
                output_all_encoded_layers=True):
        # zjm 2020/12/15: add parameter: frozen_en_layer for small dataset finetune.
        # print('[Model] Frozen_en_layer {}'.format(frozen_en_layers))
        all_encoder_layers = []
        hidden_states = input_
        for i, layer_module in enumerate(self.layer):
            if i <= frozen_en_layers - 1:
                with torch.no_grad():
                    hidden_states = layer_module(hidden_states, attention_mask)
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    BertLayer format:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
    So the final output is layernorm.
    """
    def __init__(self, config, img_dim, speech_dim, use_visual, use_speech):
        super().__init__(config)
        self.use_visual = use_visual
        self.use_speech = use_speech
        self.embeddings = UniterTextEmbeddings(config)
        if self.use_visual:
            logger.info('[INFO] Use visual modality!!!')
            self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        if self.use_speech:
            logger.info('[INFO] Use speech modality!!!')
            self.speech_embeddings = UniterSpeechEmbeddings(config, speech_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None, token_pos_tag_ids=None, 
                                token_senti_ids=None, token_utt_senti_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids, 
                                                    token_pos_tag_ids=None, 
                                                    token_senti_ids=None, 
                                                    token_utt_senti_ids=None)
        return output

    def _compute_img_embeddings(self, img_feat, img_position_ids, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        # share the embedding defined in txtEmbedding
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_position_ids,
                                     img_type_embeddings, img_masks)
        return output
    
    def _compute_speech_embeddings(self, speech_feat, speech_position_ids, speech_masks=None,
                                            speech_type_ids=None):
        if not self.use_visual or self.config.speech_visual_use_same_type:
            # logger.info('[Debug] use visual bert-type-token embedding for speech')
            speech_type_ids = torch.ones_like(speech_feat[:, :, 0].long())
            # share the embedding defined in txtEmbedding
            speech_type_embeddings = self.embeddings.token_type_embeddings(
                speech_type_ids)
        else:
            speech_type_embeddings = None
        output = self.speech_embeddings(speech_feat, speech_position_ids, \
                                        speech_type_embeddings, speech_masks)
        return output

    def _compute_img_speech_embeddings(self, img_feat, img_position_ids,
                                    speech_feat, speech_position_ids,
                                    gather_index,
                                    img_masks=None, speech_masks=None,  
                                    img_type_ids=None, speech_type_ids=None):
        img_emb = self._compute_img_embeddings(
            img_feat, img_position_ids, img_masks, img_type_ids)
        speech_emb = self._compute_speech_embeddings(
            speech_feat, speech_position_ids, speech_masks, speech_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([img_emb, speech_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_position_ids,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None,
                                    token_pos_tag_ids=None, 
                                    token_senti_ids=None, 
                                    token_utt_senti_ids=None):
        txt_emb = self._compute_txt_embeddings(
                    input_ids, position_ids, txt_type_ids, 
                        token_pos_tag_ids, token_senti_ids, token_utt_senti_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_position_ids, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def _compute_speech_txt_embeddings(self, input_ids, position_ids,
                                    speech_feat, speech_position_ids,
                                    gather_index, speech_masks=None,
                                    txt_type_ids=None, speech_type_ids=None,
                                    token_pos_tag_ids=None, 
                                    token_senti_ids=None, 
                                    token_utt_senti_ids=None):
        txt_emb = self._compute_txt_embeddings(
                            input_ids, position_ids, txt_type_ids,
                            token_pos_tag_ids, token_senti_ids, token_utt_senti_ids)
        speech_emb = self._compute_speech_embeddings(
            speech_feat, speech_position_ids, speech_masks, speech_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        # print('[Debug] before gather embedding input {}'.format(torch.cat([txt_emb, speech_emb], dim=1).shape))
        embedding_output = torch.gather(torch.cat([txt_emb, speech_emb], dim=1),
                                        dim=1, index=gather_index)
        # print('[Debug] after gather embedding_output {}'.format(embedding_output.shape))
        return embedding_output
    
    def _compute_speech_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_position_ids,
                                    speech_feat, speech_position_ids,
                                    gather_index,
                                    img_masks=None, speech_masks=None,  
                                    txt_type_ids=None, img_type_ids=None, 
                                    speech_type_ids=None,
                                    token_pos_tag_ids=None, 
                                    token_senti_ids=None, 
                                    token_utt_senti_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids,
            token_pos_tag_ids, token_senti_ids, token_utt_senti_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_position_ids, img_masks, img_type_ids)
        speech_emb = self._compute_speech_embeddings(
            speech_feat, speech_position_ids, speech_masks, speech_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb, speech_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, use_emolare_input=False, img_masks=None, speech_masks=None,
                frozen_en_layers=0, output_all_encoded_layers=False,
                txt_type_ids=None, img_type_ids=None, speech_type_ids=None):
        '''use_emolare_input; if True, the use the input type as SentiLARE
        if False: the use the input type as previous.
        '''
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_position_ids = batch['img_position_ids']
        speech_feat = batch['speech_feat']
        speech_position_ids = batch['speech_position_ids']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        if self.config.use_emolare and use_emolare_input:
            # print('[Debug] use the LARE tasks!!!')
            token_pos_tag_ids = batch['input_ids_pos']
            token_senti_ids = batch['input_ids_senti']
            token_utt_senti_ids = batch['sentence_polarity_ids']
        else:
            token_pos_tag_ids, token_senti_ids, token_utt_senti_ids,  = None, None, None

        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 当只 finetune top layers的时候, embeddings 也都要 fixed.
        # 由于在三模态的时候我们也可以做两两模态的对比，所以这里采用构建的dataloader的情况进行判断
        if frozen_en_layers > 0:
            with torch.no_grad():
                if input_ids is not None:
                    # logger.info('[Debug] the txt modality is Not None')
                    if img_feat is not None and speech_feat is None:
                        # logger.info('\t[Debug] Only the img feat Avaiable')
                        embedding_output = self._compute_img_txt_embeddings(
                            input_ids, position_ids,
                            img_feat, img_position_ids,
                            gather_index, img_masks, txt_type_ids, 
                            img_type_ids, token_pos_tag_ids, 
                            token_senti_ids, token_utt_senti_ids)
                    elif speech_feat is not None and img_feat is None:
                        # logger.info('\t[Debug] Only the speech feat Avaiable')
                        embedding_output = self._compute_speech_txt_embeddings(
                            input_ids, position_ids,
                            speech_feat, speech_position_ids,
                            gather_index, speech_masks, txt_type_ids,
                            speech_type_ids, token_pos_tag_ids, 
                            token_senti_ids, token_utt_senti_ids)
                    elif speech_feat is not None and img_feat is not None:
                        # logger.info('\t[Debug] the speech feat Avaiable and img feat Avaiable')
                        embedding_output = self._compute_speech_img_txt_embeddings(
                                input_ids, position_ids,
                                img_feat, img_position_ids,
                                speech_feat, speech_position_ids,
                                gather_index,
                                img_masks, speech_masks, 
                                txt_type_ids, img_type_ids,
                                speech_type_ids, token_pos_tag_ids, 
                                token_senti_ids, token_utt_senti_ids)
                    elif speech_feat is None and img_feat is None:
                        # 如果只包含一个模态，那么不需要gather
                        # logger.info('\t[Debug] the text feat Avaiable')
                        embedding_output = self._compute_txt_embeddings(
                            input_ids, position_ids, txt_type_ids,
                            token_pos_tag_ids, token_senti_ids, token_utt_senti_ids)
                    else:
                        logger.info('[Error] some error in UniterModel')
                        exit(0)
                else:
                    # logger.info('[Debug] the txt modality is None!!!')
                    if img_feat is not None and speech_feat is None:
                        # logger.info('\t [Debug] Only the visual feat Avaiable')
                        embedding_output = self._compute_img_embeddings(
                                    img_feat, img_position_ids,
                                    img_masks, img_type_ids, token_pos_tag_ids, 
                                    token_senti_ids, token_utt_senti_ids)
                    elif speech_feat is not None and img_feat is None:
                        # logger.info('\t[Debug] Only the speech feat Avaiable')
                        embedding_output = self._compute_speech_embeddings(
                            speech_feat, speech_position_ids,
                            speech_masks, speech_type_ids, token_pos_tag_ids, 
                            token_senti_ids, token_utt_senti_ids)
                    # add on case on not none and not None
                    else:
                        # logger.info('\t[Debug] both the visual and speech feat Avaiable')
                        embedding_output = self._compute_img_speech_embeddings(
                                img_feat, img_position_ids,
                                speech_feat, speech_position_ids,
                                gather_index,
                                img_masks, speech_masks, 
                                img_type_ids, speech_type_ids,
                                token_pos_tag_ids, 
                                token_senti_ids, token_utt_senti_ids)
        else:
            if input_ids is not None:
                # logger.info('[Debug] the txt modality is Not None')
                if img_feat is not None and speech_feat is None:
                    # logger.info('[Debug] Only the img feat Avaiable')
                    embedding_output = self._compute_img_txt_embeddings(
                        input_ids, position_ids,
                        img_feat, img_position_ids,
                        gather_index, img_masks, txt_type_ids, 
                        img_type_ids,
                        token_pos_tag_ids, 
                        token_senti_ids, token_utt_senti_ids)
                elif speech_feat is not None and img_feat is None:
                    # logger.info('[Debug] Only the speech feat Avaiable')
                    embedding_output = self._compute_speech_txt_embeddings(
                        input_ids, position_ids,
                        speech_feat, speech_position_ids,
                        gather_index, speech_masks, txt_type_ids,
                        speech_type_ids, token_pos_tag_ids, 
                        token_senti_ids, token_utt_senti_ids)
                elif speech_feat is not None and img_feat is not None:
                    # logger.info('[Debug] the speech feat Avaiable and img feat Avaiable')
                    embedding_output = self._compute_speech_img_txt_embeddings(
                            input_ids, position_ids,
                            img_feat, img_position_ids,
                            speech_feat, speech_position_ids,
                            gather_index,
                            img_masks, speech_masks, 
                            txt_type_ids, img_type_ids,
                            speech_type_ids, token_pos_tag_ids, 
                            token_senti_ids, token_utt_senti_ids)
                elif speech_feat is None and img_feat is None:
                    embedding_output = self._compute_txt_embeddings(
                            input_ids, position_ids, txt_type_ids,
                            token_pos_tag_ids, token_senti_ids, token_utt_senti_ids)
                else:
                    logger.info('[Error] some error in UniterModel')
                    exit(0)
            else:
                # logger.info('[Debug] the txt modality is None!!!')
                if img_feat is not None and speech_feat is None:
                    # logger.info('\t [Debug] Only the visual feat Avaiable')
                    embedding_output = self._compute_img_embeddings(
                                img_feat, img_position_ids,
                                img_masks, img_type_ids,
                                token_pos_tag_ids, 
                                token_senti_ids, token_utt_senti_ids)
                elif speech_feat is not None and img_feat is None:
                    # logger.info('\t[Debug] Only the speech feat Avaiable')
                    embedding_output = self._compute_speech_embeddings(
                        speech_feat, speech_position_ids,
                        speech_masks, speech_type_ids,
                        token_pos_tag_ids, 
                        token_senti_ids, token_utt_senti_ids)
                else:
                    # logger.info('\t[Debug] both the visual and speech feat Avaiable')
                    embedding_output = self._compute_img_speech_embeddings(
                            img_feat, img_position_ids,
                            speech_feat, speech_position_ids,
                            gather_index,
                            img_masks, speech_masks, 
                            img_type_ids, speech_type_ids,
                            token_pos_tag_ids, 
                            token_senti_ids, token_utt_senti_ids)
        # for model output
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            frozen_en_layers=frozen_en_layers,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers