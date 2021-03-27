import numpy as np
import torch
import json
import logging
from torch import nn
from code.uniter3flow.model.model import BertConfig, BertPreTrainedModel, BertEncoder
from code.uniter.model.layer import BertLayer, BertPooler
from apex.normalization.fused_layer_norm import FusedLayerNorm

'''
Cross-encoder model, may including 2~4 transformer layers.
用于接收三个模态的 encoder 的输出，然后做 Concat + attention-mask (音频需要有降采样，所以没有Attention Mask) 进行拼接.

'''



