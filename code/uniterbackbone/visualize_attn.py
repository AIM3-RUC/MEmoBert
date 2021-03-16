import argparse
from collections import defaultdict
import json
from os.path import join
from time import time

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


from code.uniterbackbone.model.pretrain import UniterForPretraining
from code.uniterbackbone.utils.const import IMG_LABEL_DIM
from code.denseface.hook_demo import MultiLayerFeatureExtractor
from code.uniterbackbone.pretrain import create_dataloaders

IMG_DIM = 342
# model_config = "config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_res_onlymlm.json"
model_config = "config/uniter-base-backbone_resnet.json"
checkpoint = "/data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_3tasks_faceth0.1_new_5e5_wd.01/ckpt/model_step_36000.pt"
checkpoint = torch.load(checkpoint)
# init model
model = UniterForPretraining.from_pretrained(
        model_config, checkpoint,
        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)

print(model)
# init hook
hook = MultiLayerFeatureExtractor(model, \
        [f'uniter.encoder.layer[{i}].attention.self.softmax'] for i in range(12))

# init dataset
# datast = ...
meta_loader = ...
(name, batch) = next(iter(meta_loader))
output = model(batch)
each_layer_hidden_states = hook.extract()

layer12 = each_layer_hidden_states[-1]
print(layer12.shape)