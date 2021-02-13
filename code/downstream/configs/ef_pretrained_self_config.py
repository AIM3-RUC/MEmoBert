# basic paths
# dataset_name = 'IEMOCAP'
# ft_dir_name = 'iemocap'
dataset_name = 'MSP-IMPROV'
ft_dir_name = 'msp'
result_dir = '/data7/MEmoBert/evaluation/{}/results'.format(dataset_name)
ft_dir = '/data7/MEmoBert/emobert/exp/mmfts/{}'.format(ft_dir_name)
target_dir = ft_dir

model_cfg = {
    # basic info
    'model_name':'early_fusion_multi',
    'dataset_mode': 'iemocap_pretrained', # same as msp
    'pretained_ft_type':'nomask_movies_v1_uniter_4tasks_nofinetune',
    # global training info
    'dropout_rate': 0.5,
    'modality':'VL',
    'mid_fusion_layers':'256,128',  # fusion layers
    'output_dim':4,
    'bn':False,
    'batch_size':128,
    # for learning rate
    'learning_rate':2e-4, ###*** self=2e-4, www=1e-3
    'lr_policy':'linear',
    # for training 
    'fix_lr_epoch': 20, # real fix_lr_epoch = fix_lr_epoch - warmup_epoch
    'max_epoch': 40,
    'patience':40,
    'warmup_epoch': 0,
    'warmup_decay':0.1, # warmup_learning_rate = warmup_decay * learning_rate
    'optim':'adam',
    'betas':[0.9, 0.998],  ###*** self=[0.9, 0.998], www=[0.5, 0.98]
    'grad_norm':2.0,   ###*** self=2.0, www=0.1
    'weight_decay': 0.01,
     # for different module initializers:  none / orthogonal / xavier / normal / kaiming
    'init_type': 'none',
    # visual encoer info -- lstm
    'max_visual_tokens': 50,
    'v_input_size':768,
    'v_embd_method':'maxpool', # use last mean att max
    'v_hidden_size':128, # rnn 
    # audio encoer info -- lstm
    'max_acoustic_tokens':50,
    'a_input_size':768,
    'a_embd_method':'maxpool', # use last mean att max
    'a_hidden_size':128,
    # text encoder info -- textcnn
    'max_lexical_tokens': 22,
    'l_input_size': 768,
    'l_hidden_size': 128,
}
