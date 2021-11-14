# basic paths
ft_dir = '/data7/emobert/exp/uniter3m_fts/msp/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800_nofinetune'
result_dir = '/data7/emobert/exp/evaluation/MSP/results/'
target_dir = ft_dir

model_cfg = {
    # basic info
    'model_name': 'early_fusion_multi',
    'dataset_mode': 'msp_pretrained', # use the original denseface features
    'pretained_ft_type': 'utt_baseline',
    # global training info
    'dropout_rate': 0.5,
    'modality':'VL',
    'mid_fusion_layers':'256,128',  # fusion layers
    'output_dim':4,
    'bn':False,
    'batch_size':128,
    # for learning rate
    'learning_rate':1e-3,
    'lr_policy':'linear',
    # for training
    'fix_lr_epoch': 20, # real fix_lr_epoch = fix_lr_epoch - warmup_epoch
    'max_epoch': 40,
    'patience':40,
    'warmup_epoch':0,
    'warmup_decay':0.1, # warmup_learning_rate = warmup_decay * learning_rate
    'optim':'adam',
    'betas':[0.9, 0.98],
    'grad_norm': 5.0,
    # for different module initializers:  none / orthogonal / xavier / normal / kaiming
    'init_type': 'none',
    # visual encoer info -- lstm
    'max_visual_tokens': 50,
    'v_input_size':768,
    'v_embd_method':'maxpool', # use last mean att max
    'v_hidden_size':128,  # rnn
    # visual3d encoder info resnet3d + lstm
    'v_ft_name':'openface_raw_img112',
    'v3d_img_size': 112, 
    'v3d_input_size': 512,
    'v3d_embd_method':'maxpool',
    'v3d_hidden_size': 128,
    # audio encoer info -- lstm
    'max_acoustic_tokens': 64,
    'a_ft_name': '',
    'a_input_size':768,
    'a_embd_method':'maxpool', # use last mean att max
    'a_hidden_size':128,
    # text encoder info -- textcnn, bert_base=768, bert_large=1024
    'max_text_tokens': 22,
    'l_ft_name': '',
    'l_input_size': 768,
    'l_hidden_size': 128,
}