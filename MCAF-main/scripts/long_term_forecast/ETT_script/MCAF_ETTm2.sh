#!/bin/bash

model_name=MCAF

# ============================================
# Optimized for MCAF on ETTm2 (7 variables, 15-min granularity)
# ETTm2 characteristics:
# - Same frequency as ETTm1 (15-min) but MORE CHALLENGING
# - Higher noise and irregularity than ETTm1
# - Weaker periodicity patterns
# - Requires strongest regularization among all ETT datasets
# - Benefits from robust loss functions
# ============================================

# ============================================
# ETTm2 96->96: Short-term prediction (24 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 224 \
  --d_core 56 \
  --d_ff 448 \
  --batch_size 96 \
  --learning_rate 0.0008 \
  --train_epochs 40 \
  --patience 7 \
  --lradj cosine_with_warmup \
  --warmup_epochs 5 \
  --dropout 0.15 \
  --use_norm 1 \
  --activation gelu \
  --loss huber \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm2 96->192: Medium-term prediction (48 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_core 64 \
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.0007 \
  --train_epochs 45 \
  --patience 8 \
  --lradj cosine_with_warmup \
  --warmup_epochs 6 \
  --dropout 0.2 \
  --use_norm 1 \
  --activation gelu \
  --loss huber \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm2 96->336: Long-term prediction (84 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 288 \
  --d_core 72 \
  --d_ff 576 \
  --batch_size 48 \
  --learning_rate 0.0006 \
  --train_epochs 50 \
  --patience 10 \
  --lradj cosine_with_warmup \
  --warmup_epochs 7 \
  --dropout 0.25 \
  --use_norm 1 \
  --activation gelu \
  --loss huber \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm2 96->720: Ultra-long-term prediction (180 hours / 7.5 days)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 320 \
  --d_core 80 \
  --d_ff 640 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 60 \
  --patience 12 \
  --lradj cosine_with_warmup \
  --warmup_epochs 8 \
  --dropout 0.3 \
  --use_norm 1 \
  --activation gelu \
  --loss huber \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0
