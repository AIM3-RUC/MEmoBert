# basic paths
result_dir = '/data7/MEmoBert/evaluation/MSP-IMPROV/results'
ft_dir = '/data7/MEmoBert/emobert/exp/mmfts/msp'
target_dir = ft_dir

# basic info
model_name = 'early_fusion_multi'
dataset_mode = 'iemocap_pretrained' # same as msp
pretained_ft_type = 'nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks_nofinetune'

model_cfg = {
    # global training info
    'dropout_rate': 0.5,
    'modality':'VL',
    'mid_fusion_layers':'256,128',  # fusion layers
    'output_dim':4,
    'bn':False,
    'batch_size':128,
    # for learning rate
    'learning_rate':1e-4,
    'lr_policy':'linear',
    # for training 
    'remain_epoch': 20,
    'max_epoch': 50,
    'patience':5,
    'warmup_epoch':5,
    'optim':'adamw', # 比adam好一点
    'betas':[0.9, 0.98],
    'grad_norm':2.0,
    'weight_decay': 0.01,
    # visual encoer info -- lstm
    'max_visual_tokens': 50,
    'v_input_size':768,
    'v_embd_method':'maxpool', # use last mean att max
    'v_hidden_size':128, # rnn 
    # audio encoer info -- lstm
    'a_input_size':768,
    'a_embd_method':'maxpool', # use last mean att max
    'a_hidden_size':128,
    # text encoder info -- textcnn
    'max_lexical_tokens': 22,
    'l_input_size': 768,
    'l_hidden_size': 128,
}
