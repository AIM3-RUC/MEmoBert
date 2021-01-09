# basic paths
result_dir = '/data7/MEmoBert/evaluation/IEMOCAP/results'
ft_dir = '/data7/MEmoBert/evaluation/IEMOCAP/feature'
target_dir = '/data7/MEmoBert/evaluation/IEMOCAP/target'

model_cfg = {
    # basic info
    'model_name': 'early_fusion_multi',
    'dataset_mode': 'iemocap_original', # use the original denseface features
    'pretained_ft_type': 'denseface_seetaface_iemocap_mean_std',
    # global training info
    'dropout_rate': 0.5,
    'modality':'VL',
    'mid_fusion_layers':'256,128',  # fusion layers
    'output_dim':4,
    'bn':False,
    'batch_size':128,
    # for learning rate
    'learning_rate':5e-4,
    'lr_policy':'linear',
    # for training
    'fix_lr_epoch': 20, # real fix_lr_epoch = fix_lr_epoch - warmup_epoch
    'max_epoch': 40,
    'patience':40,
    'warmup_epoch':0,
    'warmup_decay':0.1, # warmup_learning_rate = warmup_decay * learning_rate
    'optim':'adam', 
    'betas':[0.9, 0.998],
    'grad_norm': 5.0,
    # for different module initializers:  none / orthogonal / xavier / normal / kaiming
    'init_type': 'none',
    # visual encoer info -- lstm
    'max_visual_tokens': 50,
    'v_ft_name':'denseface_seetaface_iemocap_mean_std',
    'v_input_size':342,
    'v_embd_method':'maxpool', # use last mean att max
    'v_hidden_size':128, # rnn 
    # audio encoer info -- lstm
    'max_acoustic_tokens': 50,
    'a_ft_name': '',
    'a_input_size':768,
    'a_embd_method':'maxpool', # use last mean att max
    'a_hidden_size':128,
    # text encoder info -- textcnn
    'max_lexical_tokens': 22,
    'l_ft_name': 'bert',
    'l_input_size': 1024,
    'l_hidden_size': 128,
}