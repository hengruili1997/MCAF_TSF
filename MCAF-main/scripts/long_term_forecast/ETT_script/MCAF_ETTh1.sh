#!/bin/bash

model_name=MCAF

# ============================================
# Optimized for MCAF on ETTh1 (7 variables)
# Key differences from ECL:
# 1. Smaller d_model (fewer variables need less capacity)
# 2. Deeper e_layers (compensate for reduced width)
# 3. Higher learning rate (smaller dataset)
# 4. Longer training (ETTh1 is harder to fit)
# 5. Adaptive d_core ratio (d_core = d_model / 4)
# ============================================

# ============================================
# ETTh1 96->96: Short-term hourly prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_core 32 \
  --d_ff 256 \
  --batch_size 64 \
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
# ETTh1 96->192: Medium-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 160 \
  --d_core 40 \
  --d_ff 320 \
  --batch_size 48 \
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
# ETTh1 96->336: Long-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 192 \
  --d_core 48 \
  --d_ff 384 \
  --batch_size 32 \
  --learning_rate 0.0006 \
  --train_epochs 40 \
  --patience 8 \
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
# ETTh1 96->720: Ultra-long-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 224 \
  --d_core 56 \
  --d_ff 448 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --train_epochs 50 \
  --patience 10 \
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
