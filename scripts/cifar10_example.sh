#!/bin/bash	
CUDA_VISIBLE_DEVICES=0 python train_iter.py \
--model BigGAN_MPCC \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 300 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 --P_lr 6e-4 \
--dataset C10 \
--hier --is_encoder \
--is_enc_shared \
--G_shared \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--prior_type GMM \
--G_ch 96 --D_ch 96 \
--num_E_steps 4 \
--is_loss3 0.01 \
--concat \
--test_every 5000 --save_every 2000 --num_best_copies 3 --num_save_copies 2 --seed 0
