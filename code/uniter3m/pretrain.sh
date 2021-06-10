export PYTHONPATH=/data7/MEmoBert

# case0: only text running on gpu0  use the different init
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_text_1tasks_emocls.json \
#         --checkpoint /data7/emobert/exp/text_emo_model/all_5corpus_emo5_bert_base_lr2e-5_bs32/ckpt/uniter_init_5corpus_emo5.pt \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_initby5corpus_emo5_text_1tasks_lr5e5_bs512_faceth0.5

## case1: visual + text running on gpu0
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5

## case2: wav2vec + text running on a100, ---Going
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_text_3tasks_mlmitmmsm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30\
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 100000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs1024_train10w

## case2.1: wav2vec + text running on a100, ---Going
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_3tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_emocls_lr5e5_bs512

## case3: wav2vec + text + visual running on gpu2
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1_lr5e5_bs512_faceth0.5


# case4: text + visual + sentiword-emo running on gpu3
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_5tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword

# case4.1: text + visual + emocls running on gpu3 
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emocls_vstype2_lr5e5_bs512_faceth0.5

## case4.2: text + visual, diff emo cls (logits-label) task with itm + emoclsselected
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_emoclsselected.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emoclsselected_vstype2_lr5e5_bs1024

## case4.3: text + visual, diff emo cls (logits-label) task with itm + emoclsselected
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_emoclsselected_corpusemo4.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft_emo4.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_emoclsselected_corpusemo4_vstype2_lr5e5_bs1024

# case5: text + wav2vec + sentiword-emo running on gpu5
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_wav2vec_3tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 128 --val_batch_size 128 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_4tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword

# 之前的visual和speech采用相同的type-embedding，但是采用的 position embeeding 却不同
# 所以这里修改一下，加一个给不同的模态加不同的 type embedding. 只需要将speech_visual_use_same_type=false就可以
# case6: text + wav2vec + visual, diff type-embedding running on a100
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 100000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs800_train10w

## case6.1: wav2vec + text + visual running on a100, +melm
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask_difftype.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 80000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs1024_train10w_sentiword

## case6.2: text + visual, diff emo cls (logits-label) task with itm + emoclsselected
# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emoclsselected.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 80 --val_batch_size 80 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_emoclsselected_vstype2_lr5e5_bs1024

# ## case6.3: text + visual, diff emo cls (logits-label) task with itm + emoclsselected
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emoclsselected_corpusemo4.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft_emo4.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 80 --val_batch_size 80 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_emoclsselected_corpusemo4_vstype2_lr5e5_bs1024

# 将 itm 任务替换为 vtm 和 stm 两个任务, 这样可以做下游的两模态的任务
# case7: text + wav2vec + visual, diff type-embedding running on leo --training
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_6tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_6tasks_vstype2_lr5e5_bs512_faceth0.5

# 增加除 itm 之外的 vtm 和 stm 两个任务, 共7个预训练任务了
# case7: text + wav2vec + visual, diff type-embedding running on a100
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_7tasks.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 100000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_7tasks_vstype2_lr5e5_bs1024_train10w

### case8: text + wav2vec + visual, diff emo cls task no itm task running on a100
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type probs \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabel.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsSoft_noitm_vstype2_lr5e5_bs1024

### case8.1: text + wav2vec + visual, diff emo cls (hard-label) task with itm task running on a100
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type hard \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelHard.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsHard_withitm_vstype2_lr5e5_bs1024

### case8.2: text + wav2vec + visual, diff emo cls (logits-label) task with itm task running on a100
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type logits --emocls_temperture 2.0 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelLogits.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsLogits_withitm_vstype2_lr5e5_bs1024

### case9: text + wav2vec + visual, diff emo cls task with itm task running on a100
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type soft \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5_bs1024

### case9.1: text + wav2vec + visual, diff emo cls (hard-label) task with itm task running on a100
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type hard \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelHard.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsHard_withitm_vstype2_lr5e5_bs1024

### case9.2: text + wav2vec + visual, diff emo cls (logits-label) task with itm task running on a100
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emocls_type logits --emocls_temperture 2.0 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelLogits.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsLogits_withitm_vstype2_lr5e5_bs1024


### case9.3: text + wav2vec + visual, diff emo cls (logits-label) task with itm task running on a100
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --emolare_LStask_ratio 0.2 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_lare.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclslare0.2_noitm_vstype2_lr5e5_bs1024

### case9.4: text + wav2vec + visual, diff emo cls (logits-label) task with itm--all_corpus_emo5
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls_corpusemo5.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 50000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emocls_corpusemo5_vstype2_lr5e5_bs1024

### case9.5: text + wav2vec + visual, diff emo cls (logits-label) task with itm task --all_corpus_emo5
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls_corpusemo4.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft_emo4.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 50000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emocls_corpusemo4_vstype2_lr5e5_bs1024

### case9.6: text + wav2vec + visual, diff emo cls (logits-label) task with itm task --all_corpus_emo7
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_emocls_corpusemo7.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft_emo7.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 50000 --warmup_steps 5000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emocls_corpusemo7_vstype2_lr5e5_bs1024


### case10.2: text + wav2vec + visual, diff emo cls (probs + only important word) running on a100
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_nomlm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_nomlm_withmelm_nomultitask_vstype2_lr5e5_bs1024

### case11.1: text + wav2vec + visual, diff emo cls (probs + eitm task) running on a100
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsSoft_eitm_vstype2_lr5e5_bs1024

### case11.2: text + wav2vec + visual, diff emo cls (probs + eitm task) running on a100
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_eitm_vstype2_lr5e5_bs1024


### case11.1: text + wav2vec + visual, diff emo cls (probs + eitm task) running on a100
# CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --use_total_eitm \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_4tasks_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 25 --val_batch_size 25 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 30 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsSoft_Teitm_vstype2_lr5e5_bs1024

### case11.2: text + wav2vec + visual, diff emo cls (probs + eitm task) running on a100
# CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --use_total_eitm \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_eitm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 256 --val_batch_size 256 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_Teitm_vstype2_lr5e5_bs1024

### case12.1: text + wav2vec + visual, emo cls (itm + mlm + probs + merm two task) running on a100
# CUDA_VISIBLE_DEVICES=6 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_3tasks_merm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_3tasks_merm_emoclsSoft_vstype2_lr5e5_bs1024

### case12.2: text + wav2vec + visual, emo cls (itm + mlm + mrfr + mrckl + probs + merm two task) running on a100
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_merm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_merm_emoclsSoft_vstype2_lr5e5_bs1024

### case13: text + wav2vec + visual, emo cls (itm + mlm + mrfr + mrckl + probs + merm two task) running on a100
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_speech --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_merm_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --max_txt_len 30 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 3000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_merm_emoclsSoft_vstype2_lr5e5_bs1024