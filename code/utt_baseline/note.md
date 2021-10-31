参数调整:
1. 根据WF1和F1保留模型，最终结果根据WF1的结果
2. 模型维度一般 512/256 就可以 --mid_fusion_layers '256,128' # 单模态 --mid_fusion_layers '512,256'  # 多模态


## A-comparE_norm
2021-08-27 09:09:09,669 - 2021-08-27-09.04.59 - INFO - [Val] result WA: 0.3839 UAR 0.2406 F1 0.2349
2021-08-27 09:09:10,823 - 2021-08-27-09.04.59 - INFO - [Tst] result WA: 0.3632 UAR 0.2112 F1 0.2033

## A-IS10_norm(256)
2021-08-27 09:18:03,477 - 2021-08-27-09.16.12 - INFO - Loading best model found on val set: epoch-38
2021-08-27 09:18:03,663 - 2021-08-27-09.16.12 - INFO - [Val] result WA: 0.4218 UAR 0.2276 F1 0.2262
2021-08-27 09:18:03,902 - 2021-08-27-09.16.12 - INFO - [Tst] result WA: 0.4535 UAR 0.2398 F1 0.2429

# A_lr0.001_dp0.5_bnFalse_AIS10_norm256_Vdenseface256_Lbert_base_chinese256_F256,128_run1_self
2021-08-27 09:18:03,477 - 2021-08-27-09.16.12 - INFO - Loading best model found on val set: epoch-38
2021-08-27 09:18:03,663 - 2021-08-27-09.16.12 - INFO - [Val] result WA: 0.4218 UAR 0.2276 F1 0.2262
2021-08-27 09:18:03,902 - 2021-08-27-09.16.12 - INFO - [Tst] result WA: 0.4535 UAR 0.2398 F1 0.2429

## A-wav2vec_zh > comparE_norm
2021-08-26 19:42:42,351 - 2021-08-26-17.54.29 - INFO - [Val] result WA: 0.3988 UAR 0.2446 F1 0.2476
2021-08-26 19:42:44,659 - 2021-08-26-17.54.29 - INFO - [Tst] result WA: 0.4006 UAR 0.2336 F1 0.2363

# A-Wav2vec-zh（512）要比256维度的要好
2021-08-27 03:12:19,937 - 2021-08-27-03.06.50 - INFO - Loading best model found on val set: epoch-9
2021-08-27 03:12:21,399 - 2021-08-27-03.06.50 - INFO - [Val] result WA: 0.4424 UAR 0.2605 F1 0.2647
2021-08-27 03:12:23,569 - 2021-08-27-03.06.50 - INFO - [Tst] result WA: 0.4275 UAR 0.2432 F1 0.2469

# A-sent_last-Wav2vec-zh（256）
2021-08-27 14:18:50,333 - 2021-08-27-14.16.15 - INFO - Loading best model found on val set: epoch-21
2021-08-27 14:18:50,808 - 2021-08-27-14.16.15 - INFO - [Val] result WA: 0.4406 UAR 0.2114 F1 0.1890
2021-08-27 14:18:51,542 - 2021-08-27-14.16.15 - INFO - [Tst] result WA: 0.4692 UAR 0.2169 F1 0.1983

# A-sent_avg-Wav2vec-zh（256）
Loading best model found on val set: epoch-39
[Val] result WA: 0.4885 UAR 0.2526 F1 0.2440
[Tst] result WA: 0.4901 UAR 0.2365 F1 0.2269

# A-sent_wav2vec_zh2chmed2e5last(256) 
2021-08-28 15:49:55,193 - 2021-08-28-15.46.39 - INFO - Loading best model found on val set: epoch-14
2021-08-28 15:49:55,608 - 2021-08-28-15.46.39 - INFO - [Val] result WA: 0.4973 UAR 0.3262 F1 0.3382
2021-08-28 15:49:56,732 - 2021-08-28-15.46.39 - INFO - [Tst] result WA: 0.4677 UAR 0.2847 F1 0.2913
# A-sent_wav2vec_zh2chmed2e5last(512) -- use this
2021-08-28 15:51:51,824 - 2021-08-28-15.47.05 - INFO - Loading best model found on val set: epoch-12
2021-08-28 15:51:52,982 - 2021-08-28-15.47.05 - INFO - [Val] result WA: 0.4970 UAR 0.3322 F1 0.3457
2021-08-28 15:51:54,989 - 2021-08-28-15.47.05 - INFO - [Tst] result WA: 0.4725 UAR 0.2926 F1 0.2990

## L-Bert-base-Chinese
Loading best model found on val set: epoch-11
[Val] result WA: 0.4495 UAR 0.2821 F1 0.2801
[Tst] result WA: 0.4661 UAR 0.2555 F1 0.2650

## L-RoBert-base-wwm-Chinese
2021-08-27 09:33:27,448 - 2021-08-27-09.30.48 - INFO - Loading best model found on val set: epoch-16
2021-08-27 09:33:28,051 - 2021-08-27-09.30.48 - INFO - [Val] result WA: 0.4282 UAR 0.2913 F1 0.2878
2021-08-27 09:33:28,939 - 2021-08-27-09.30.48 - INFO - [Tst] result WA: 0.4382 UAR 0.2648 F1 0.2743

## L-sent_avg_Bert-base-Chinese
2021-08-27 12:49:16,662 - 2021-08-27-12.48.06 - INFO - Loading best model found on val set: epoch-6
2021-08-27 12:49:17,112 - 2021-08-27-12.48.06 - INFO - [Val] result WA: 0.4474 UAR 0.2731 F1 0.2578
2021-08-27 12:49:17,777 - 2021-08-27-12.48.06 - INFO - [Tst] result WA: 0.4704 UAR 0.2420 F1 0.2383

## L-sent_cls_Bert-base-Chinese(cls要比avg要好)
2021-08-27 12:49:35,626 - 2021-08-27-12.48.19 - INFO - Loading best model found on val set: epoch-7
2021-08-27 12:49:35,978 - 2021-08-27-12.48.19 - INFO - [Val] result WA: 0.4498 UAR 0.2903 F1 0.2642
2021-08-27 12:49:36,447 - 2021-08-27-12.48.19 - INFO - [Tst] result WA: 0.4782 UAR 0.2698 F1 0.2649

## L-sent_avg_RoBert-base-wwm-Chinese
2021-08-27 12:59:06,496 - 2021-08-27-12.57.33 - INFO - Loading best model found on val set: epoch-13
2021-08-27 12:59:06,790 - 2021-08-27-12.57.33 - INFO - [Val] result WA: 0.4587 UAR 0.2793 F1 0.2680
2021-08-27 12:59:07,211 - 2021-08-27-12.57.33 - INFO - [Tst] result WA: 0.4796 UAR 0.2567 F1 0.2585

## L-sent_cls_RoBert-base-wwm-Chinese
2021-08-27 12:56:13,077 - 2021-08-27-12.55.12 - INFO - Loading best model found on val set: epoch-6
2021-08-27 12:56:13,500 - 2021-08-27-12.55.12 - INFO - [Val] result WA: 0.4555 UAR 0.2768 F1 0.2629
2021-08-27 12:56:14,045 - 2021-08-27-12.55.12 - INFO - [Tst] result WA: 0.4651 UAR 0.2470 F1 0.2401

## L-sent_cls_robert_wwm_base_chinese4chmed(Finetune之后的效果不错，跟直接finetune的结果基本一致)
Save model at 8 epoch
Loading best model found on val set: epoch-3
[Val] result WA: 0.4615 UAR 0.3433 F1 0.3282
[Tst] result WA: 0.4563 UAR 0.3150 F1 0.3143

## L-sent_cls_bert_base_chinese4chmed
Loading best model found on val set: epoch-6
[Val] result WA: 0.4516 UAR 0.3263 F1 0.3258
[Tst] result WA: 0.4468 UAR 0.2893 F1 0.3010

## V-DenseFace
2021-08-25 19:26:37,008 - 2021-08-25-19.17.54 - INFO - Loading best model found on val set: epoch-17
2021-08-25 19:26:38,583 - 2021-08-25-19.17.54 - INFO - [Val] result WA: 0.4573 UAR 0.2671 F1 0.2640
2021-08-25 19:26:40,717 - 2021-08-25-19.17.54 - INFO - [Tst] result WA: 0.4268 UAR 0.2503 F1 0.2368
## V-AffectDenseFace(2e-5)
Loading best model found on val set: epoch-19
[Warning] error msg []
[Val] result WA: 0.4764 UAR 0.2763 F1 0.2747
[Tst] result WA: 0.4506 UAR 0.2505 F1 0.2431

## V-sent_avg_DenseFace
Loading best model found on val set: epoch-8
[Val] result WA: 0.4559 UAR 0.2424 F1 0.2279
[Tst] result WA: 0.4511 UAR 0.2315 F1 0.2083

## V-DenseFace + L-Bert-base-Chinese
2021-08-26 19:39:13,690 - 2021-08-26-17.53.04 - INFO - Loading best model found on val set: epoch-13
2021-08-26 19:39:25,922 - 2021-08-26-17.53.04 - INFO - [Val] result WA: 0.5101 UAR 0.3762 F1 0.3723
2021-08-26 19:39:43,091 - 2021-08-26-17.53.04 - INFO - [Tst] result WA: 0.4892 UAR 0.3459 F1 0.3452

## V-Denseface + L-RoBert-base-WWM-Chinese
2021-08-27 09:47:10,777 - 2021-08-27-09.39.08 - INFO - Loading best model found on val set: epoch-12
2021-08-27 09:47:12,085 - 2021-08-27-09.39.08 - INFO - [Val] result WA: 0.5062 UAR 0.3783 F1 0.3741
2021-08-27 09:47:13,814 - 2021-08-27-09.39.08 - INFO - [Tst] result WA: 0.4618 UAR 0.3410 F1 0.3329

## A-wav2vec-zh + L-Bert-base-Chinese
# AL_lr0.001_dp0.5_bnFalse_Awav2vec_zh256_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 01:55:43,238 - 2021-08-27-01.46.43 - INFO - Loading best model found on val set: epoch-8
2021-08-27 01:55:46,156 - 2021-08-27-01.46.43 - INFO - [Val] result WA: 0.4952 UAR 0.3101 F1 0.3115
2021-08-27 01:55:50,469 - 2021-08-27-01.46.43 - INFO - [Tst] result WA: 0.4742 UAR 0.2773 F1 0.2734
# AL_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:39:53,703 - 2021-08-27-02.28.51 - INFO - Loading best model found on val set: epoch-8
2021-08-27 02:39:55,755 - 2021-08-27-02.28.51 - INFO - [Val] result WA: 0.4665 UAR 0.3288 F1 0.3220
2021-08-27 02:39:58,758 - 2021-08-27-02.28.51 - INFO - [Tst] result WA: 0.4563 UAR 0.3011 F1 0.3024

## A-IS10_norm + L-Bert-base-Chinese
2021-08-27 09:29:53,772 - 2021-08-27-09.26.03 - INFO - Loading best model found on val set: epoch-13
2021-08-27 09:29:54,539 - 2021-08-27-09.26.03 - INFO - [Val] result WA: 0.5034 UAR 0.3495 F1 0.3583
2021-08-27 09:29:55,652 - 2021-08-27-09.26.03 - INFO - [Tst] result WA: 0.4944 UAR 0.3054 F1 0.3192

## A-IS10_norm + L-RoBert-base-WWM-Chinese
2021-08-27 09:45:47,042 - 2021-08-27-09.39.13 - INFO - Loading best model found on val set: epoch-15
2021-08-27 09:45:49,451 - 2021-08-27-09.39.13 - INFO - [Val] result WA: 0.5080 UAR 0.3448 F1 0.3508
2021-08-27 09:45:52,497 - 2021-08-27-09.39.13 - INFO - [Tst] result WA: 0.4858 UAR 0.3102 F1 0.3227

## A-comparE_norm + V-Denseface
2021-08-26 19:30:55,396 - 2021-08-26-17.53.12 - INFO - Loading best model found on val set: epoch-15
2021-08-26 19:31:26,475 - 2021-08-26-17.53.12 - INFO - [Val] result WA: 0.4747 UAR 0.3090 F1 0.3056
2021-08-26 19:32:04,459 - 2021-08-26-17.53.12 - INFO - [Tst] result WA: 0.4385 UAR 0.2717 F1 0.2623

 ## A-wav2vec-zh + V-Denseface
2021-08-27 01:54:31,011 - 2021-08-27-01.47.10 - INFO - Loading best model found on val set: epoch-11
2021-08-27 01:54:32,971 - 2021-08-27-01.47.10 - INFO - [Val] result WA: 0.4467 UAR 0.3059 F1 0.3226
2021-08-27 01:54:36,122 - 2021-08-27-01.47.10 - INFO - [Tst] result WA: 0.4142 UAR 0.2666 F1 0.2727
# AV_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:39:40,424 - 2021-08-27-02.29.27 - INFO - Loading best model found on val set: epoch-11
2021-08-27 02:39:43,841 - 2021-08-27-02.29.27 - INFO - [Val] result WA: 0.4481 UAR 0.3081 F1 0.3137
2021-08-27 02:39:48,918 - 2021-08-27-02.29.27 - INFO - [Tst] result WA: 0.4297 UAR 0.2825 F1 0.2818

## A-IS10_norm + V-Denseface
2021-08-27 09:29:11,856 - 2021-08-27-09.26.08 - INFO - Loading best model found on val set: epoch-13
2021-08-27 09:29:12,703 - 2021-08-27-09.26.08 - INFO - [Val] result WA: 0.5218 UAR 0.3124 F1 0.3121
2021-08-27 09:29:13,810 - 2021-08-27-09.26.08 - INFO - [Tst] result WA: 0.4989 UAR 0.2862 F1 0.2819


## A-comparE_norm + V-Denseface + L-Bert-base-Chinese
 2021-08-26 19:42:45,293 - 2021-08-26-17.52.46 - INFO - Loading best model found on val set: epoch-12
2021-08-26 19:42:47,149 - 2021-08-26-17.52.46 - INFO - [Val] result WA: 0.5310 UAR 0.3756 F1 0.3842
2021-08-26 19:42:49,738 - 2021-08-26-17.52.46 - INFO - [Tst] result WA: 0.4942 UAR 0.3255 F1 0.3297

## A-wav2vec-zh + V-Denseface + L-Bert-base-Chinese
2021-08-27 01:54:47,458 - 2021-08-27-01.47.43 - INFO - Loading best model found on val set: epoch-7
2021-08-27 01:54:49,489 - 2021-08-27-01.47.43 - INFO - [Val] result WA: 0.5463 UAR 0.3616 F1 0.3792
2021-08-27 01:54:52,590 - 2021-08-27-01.47.43 - INFO - [Tst] result WA: 0.5165 UAR 0.3148 F1 0.3203
# AVL_lr0.001_dp0.5_bnFalse_Awav2vec_zh512_Vdenseface256_Lbert_base_chinese256_F512,256_run1_self
2021-08-27 02:40:12,251 - 2021-08-27-02.29.53 - INFO - Loading best model found on val set: epoch-8
2021-08-27 02:40:14,791 - 2021-08-27-02.29.53 - INFO - [Val] result WA: 0.5232 UAR 0.3786 F1 0.3711
2021-08-27 02:40:18,855 - 2021-08-27-02.29.53 - INFO - [Tst] result WA: 0.4620 UAR 0.3427 F1 0.3279

## A-IS10_norm + V-Denseface + L-Bert-base-Chinese(三个模态的终于有提升了）
2021-08-27 09:32:20,340 - 2021-08-27-09.26.23 - INFO - Loading best model found on val set: epoch-15
2021-08-27 09:32:21,445 - 2021-08-27-09.26.23 - INFO - [Val] result WA: 0.5452 UAR 0.4048 F1 0.4158
2021-08-27 09:32:22,993 - 2021-08-27-09.26.23 - INFO - [Tst] result WA: 0.5108 UAR 0.3517 F1 0.3645

## A-IS10_norm + V-Denseface + L-RoBert-base-WWM-Chinese (robert和bert特征差不多)
2021-08-27 09:47:26,806 - 2021-08-27-09.39.25 - INFO - Loading best model found on val set: epoch-13
2021-08-27 09:47:27,796 - 2021-08-27-09.39.25 - INFO - [Val] result WA: 0.5650 UAR 0.3975 F1 0.4132
2021-08-27 09:47:29,335 - 2021-08-27-09.39.25 - INFO - [Tst] result WA: 0.5242 UAR 0.3431 F1 0.3555

## A-sent_avg_wav2vec_zh + V-sent_avg_DenseFace + L-sent_cls_bert_base_chinese
[Val] result WA: 0.5569 UAR 0.3682 F1 0.3874
[Tst] result WA: 0.5280 UAR 0.3188 F1 0.3284

## A-sent_avg_wav2vec_zh + L-sent_cls_bert_base_chinese
Loading best model found on val set: epoch-11
[Val] result WA: 0.5151 UAR 0.3405 F1 0.3348
[Tst] result WA: 0.5099 UAR 0.3005 F1 0.3087

## V-sent_avg_DenseFace + L-sent_cls_bert_base_chinese
[Val] result WA: 0.5030 UAR 0.3481 F1 0.3531
[Tst] result WA: 0.4908 UAR 0.3126 F1 0.3164

## A-IS10_norm + V-sent_avg_DenseFace + L-sent_cls_robert_wwm_base_chinese4chmed (差不多也能取得最好的结果)
Loading best model found on val set: epoch-7
[Val] result WA: 0.5285 UAR 0.3888 F1 0.3950
[Tst] result WA: 0.5054 UAR 0.3550 F1 0.3696

## A-IS10_norm + V-DenseFace + L-sent_cls_robert_wwm_base_chinese4chmed
Loading best model found on val set: epoch-14
[Val] result WA: 0.5353 UAR 0.4040 F1 0.4048
[Tst] result WA: 0.5056 UAR 0.3778 F1 0.3808

## A-IS10_norm + L-sent_cls_robert_wwm_base_chinese4chmed（IS10表现要更好一些）
[Val] result WA: 0.4963 UAR 0.3564 F1 0.3527
[Tst] result WA: 0.4942 UAR 0.3433 F1 0.3500

## A-sent_avg_wav2vec_zh + L-sent_cls_robert_wwm_base_chinese4chmed
[Val] result WA: 0.4874 UAR 0.3603 F1 0.3553
[Tst] result WA: 0.4623 UAR 0.3261 F1 0.3312

## V-sent_avg_DenseFace + L-sent_cls_robert_wwm_base_chinese4chmed
Loading best model found on val set: epoch-7
[Val] result WA: 0.4906 UAR 0.3735 F1 0.3735
[Tst] result WA: 0.4675 UAR 0.3564 F1 0.3585


## A-sent_wav2vec_zh2chmed2e5last + V-sent_avg_DenseFace + L-sent_cls_robert_wwm_base_chinese4chmed
当特征都是high-level特征之后，学习率可以小一点
AVL_lr0.0002_dp0.5_bnFalse_Asent_wav2vec_zh2chmed2e5last256_Vsent_avg_denseface256_Lsent_cls_robert_wwm_base_chinese4chmed256_F512,256_run1_self
Loading best model found on val set: epoch-6
[Val] result WA: 0.5204 UAR 0.3859 F1 0.3945
[Tst] result WA: 0.4846 UAR 0.3489 F1 0.3560

## A-IS10_norm + V-sent_avg_DenseFace
Loading best model found on val set: epoch-14
[Val] result WA: 0.5186 UAR 0.3198 F1 0.3201
[Tst] result WA: 0.4849 UAR 0.2908 F1 0.2837

## A-sent_wav2vec_zh2chmed2e5last + V-sent_avg_DenseFace
Loading best model found on val set: epoch-11
[Val] result WA: 0.4942 UAR 0.3334 F1 0.3462
[Tst] result WA: 0.4704 UAR 0.3000 F1 0.3077
## A-sent_wav2vec_zh2chmed2e5last + L-sent_cls_robert_wwm_base_chinese4chmed
Loading best model found on val set: epoch-4
[Val] result WA: 0.5165 UAR 0.3782 F1 0.3772
[Tst] result WA: 0.4880 UAR 0.3507 F1 0.3521
