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

import numpy as np
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm

from code.uniterbackbone.model.layer import BertLayer, BertPooler
from torch import tensor
from code.denseface.model.dense_net import DenseNetEncoder
from code.denseface.model.vggnet import VggNetEncoder
from code.denseface.model.resnet import ResNetEncoder

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
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        # build position vocab embeddings = 512
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        ## add by jinming: add another emotion word type embedding
        try:
            if config.melm_type_emo_size:
                print("initialize the emo type embeddings!!!")
                self.emo_type_embeddings = nn.Embedding(config.melm_type_emo_size,
                                                        config.hidden_size)
        except:
            print("[Warning] Donot use emo type embeddings")
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None, emo_type_ids=None):
        '''
        emo_type_ids: the emotion types of the input ids
        batch-data
        '''
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)

        ## Jinming: add another emotion word type embedding
        if emo_type_ids is not None:
            # print('[Debug] verify the emo type embeddig is work or not')
            emo_type_embeddings = self.emo_type_embeddings(emo_type_ids)
            embeddings = embeddings + emo_type_embeddings
    
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.config = config
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # Jinming add for joint training denseface backbone
        if self.config.joint_face_backbone is True:
            if self.config.backbone_type == 'resnet':
                from code.denseface.config.res_fer import model_cfg as backbone_config
                self.face_encoder = ResNetEncoder(**backbone_config)
            elif self.config.backbone_type == 'densenet':
                from code.denseface.config.dense_fer import model_cfg as backbone_config
                self.face_encoder = DenseNetEncoder(**backbone_config)
            elif self.config.backbone_type == 'vggnet':
                from code.denseface.config.vgg_fer import model_cfg as backbone_config
                self.face_encoder = VggNetEncoder(**backbone_config)
            else:
                print('[Error] backbone type {}'.format(self.config.backbone_type))
            if not self.config.face_from_scratch:
                print('[Debug] Train the face backbone from {}!!!!'.format(self.config.face_checkpoint))
                state_dict = torch.load(self.config.face_checkpoint)
                for key in list(state_dict.keys()):
                    if 'classifier' in key or 'features' not in key:
                        del state_dict[key]
                # print('[Debug]****** densenet original layer weights')
                # print(state_dict['features.denseblock2.denselayer10.conv1.weight'])
                # print(state_dict['features.denseblock3.denselayer10.conv1.weight'])
                # print(list(state_dict.keys())[:20])
                # print(list(self.face_encoder.state_dict().keys())[:20])
                self.face_encoder.load_state_dict(state_dict)
            else:
                print('[Debug] Train the face backbone from scratch!!!!')

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_position_ids, img_type_embeddings, img_masks=None):
        # print('[Debug]****** traning layer weights')
        # print(self.face_encoder.state_dict()['features.denseblock2.denselayer10.conv1.weight'])
        # print(self.face_encoder.state_dict()['features.denseblock3.denselayer10.conv1.weight'])
        # Jinming add for joint training denseface backbone
        if self.config.joint_face_backbone is True:
            # 输入的是原始的图片信息, 
            new_img_feat = []
            # print('[Debug] Raw images {}'.format(img_feat.shape))
            for raw_img_4video in img_feat:
                raw_img_ft = self.face_encoder.forward(raw_img_4video)
                # print('[Debug]raw_img_ft {}'.format(raw_img_ft.shape))
                new_img_feat.append(raw_img_ft)
            img_feat = torch.stack(new_img_feat)
            # print("[Debug] densenet output {}".format(img_feat.shape))

        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask
        
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        position_embeddings = self.position_embeddings(img_position_ids)
        # print('transformed_im {} position_embeddings {} img_type_embeddings {}'.format(
        #     transformed_im.size(), position_embeddings.size(), img_type_embeddings.size()
        # ))
        embeddings = transformed_im + position_embeddings + img_type_embeddings
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
        # print('[Model]Frozen_en_layer {}'.format(frozen_en_layers))
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
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None, emo_type_ids=None):
        # Jinming: add emo_type_ids interface
        output = self.embeddings(input_ids, position_ids, txt_type_ids, emo_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_position_ids, img_masks=None,
                                img_type_ids=None):
        
        # Jinming add: if img_feat is raw images, the img_type_ids is 
        if img_type_ids is None:
            if len(list(img_feat.shape)) == 4:
                img_type_ids = torch.ones_like(img_feat[:, :, 0, 0].long())
            elif len(list(img_feat.shape)) == 3:
                img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
            else:
                print('[Error] The input image feature shape is error {}'.format(img_feat.shape))
        # share the embedding defined in txtEmbedding
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_position_ids,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_position_ids,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None, 
                                    emo_type_ids=None):
        # Jinming: add emo_type_ids interface
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids, emo_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_position_ids, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, img_masks=None,
                frozen_en_layers=0,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        # Jinming add note: if img_feat are raw images, 
        # then the img_feat with shape (batchsize, max-len, img-dim, img-dim)
        img_feat = batch['img_feat']
        img_position_ids = batch['img_position_ids']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        # Jinming: add emo_type_ids interface
        if batch.get('emo_type_ids') is not None:
            emo_type_ids = batch['emo_type_ids']
        else:
            emo_type_ids = None

        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_position_ids, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids, emo_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_position_ids,
                gather_index, img_masks, txt_type_ids, 
                img_type_ids, emo_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            frozen_en_layers=frozen_en_layers,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers