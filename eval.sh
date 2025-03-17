#!/bin/bash

for N in {1..10}
do
    CUDA_VISIBLE_DEVICES=2 python enhancement.py \
        --test_dir /data/dataset/VCTK_corpus/ \
        --enhanced_dir VB_DMD_FROM_WSJ0_CHIME_3_STORM_FLOWSE_N_$N \
        --mode storm \
        --N $N \
        --ckpt /workspace/storm_flowse/.logs/mode=regen-joint-training_ode=FLOWMATCHING_score=ncsnpp_denoiser=ncsnpp_condition=both_data=wsj0_high_ch=1/version_2/checkpoints/epoch=314.ckpt
done
