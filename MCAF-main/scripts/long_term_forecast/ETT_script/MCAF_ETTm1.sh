#!/bin/bash

model_name=MCAF

# ============================================
# Optimized for MCAF on ETTm1 (7 variables, 15-min granularity)
# ETTm1 characteristics:
# - 4x higher frequency than ETTh (15min vs 1hour)
# - Stronger short-term dependencies
# - More data points (69680 vs 17420)
# - Richer high-frequency patterns
# - Benefits from frequency-domain modeling
# ============================================

# ============================================
# ETTm1 96->96: Short-term prediction (24 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 192 \
  --d_core 48 \
  --d_ff 384 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 5 \
  --lradj cosine_with_warmup \
  --warmup_epochs 3 \
  --dropout 0.05 \
  --use_norm 1 \
  --activation gelu \
  --loss mse \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm1 96->192: Medium-term prediction (48 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 224 \
  --d_core 56 \
  --d_ff 448 \
  --batch_size 96 \
  --learning_rate 0.0008 \
  --train_epochs 35 \
  --patience 6 \
  --lradj cosine_with_warmup \
  --warmup_epochs 4 \
  --dropout 0.1 \
  --use_norm 1 \
  --activation gelu \
  --loss mse \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm1 96->336: Long-term prediction (84 hours)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_core 64 \
  --d_ff 512 \
  --batch_size 64 \
  --learning_rate 0.0006 \
  --train_epochs 40 \
  --patience 7 \
  --lradj cosine_with_warmup \
  --warmup_epochs 5 \
  --dropout 0.15 \
  --use_norm 1 \
  --activation gelu \
  --loss mse \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ETTm1 96->720: Ultra-long-term prediction (180 hours / 7.5 days)
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 288 \
  --d_core 72 \
  --d_ff 576 \
  --batch_size 32 \
  --learning_rate 0.0005 \
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
