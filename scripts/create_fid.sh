#!/bin/bash	
CUDA_VISIBLE_DEVICES=1 python fid_p.py \
--fid_stats cifar10/fid_stats_cifar10_train.npz \
--experiment_root weights \
--experiment_name BigGAN_C10_BigGAN_MPCC_seed5_Gch96_Dch96_bs50_z128_nDs4_nEs4_Glr2.0e-04_Dlr2.0e-04_Plr6.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_Gshared_hier_ema_pGMM_is_E_std0.50_L3_0.010_cpc1
