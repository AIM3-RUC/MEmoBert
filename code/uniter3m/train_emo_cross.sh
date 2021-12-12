source deactivate base

export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1


################# Part1.1: Directly Training (source IEMOCAP and Target MSP)################################################### 
################# Part2.1: Directly+BERT Training (source IEMOCAP and Target MSP)################################################### 
################# Part3.1: MEmoBERT+Finetune (source IEMOCAP and Target MSP )################################################### 
################# Part4.1: MEmoBERT+Prompt (source IEMOCAP and Target MSP )################################################### 
################# Part1.2: Directly Training (source MSP and Target IEMOCAP)################################################### 
################# Part2.2: Directly+BERT Training (source  MSP and Target IEMOCAP)################################################### 
################# Part3.2: MEmoBERT+Finetune (source MSP and Target IEMOCAP)################################################### 
################# Part4.2: MEmoBERT+Prompt (source MSP and Target IEMOCAP)################################################### 


################# Part1.1: Directly Training (source IEMOCAP and Target MSP)###################################################
source_corpus_name='iemocap'
source_corpus_name_L='IEMOCAP'
target_corpus_name='msp'
target_corpus_name_L='MSP'
