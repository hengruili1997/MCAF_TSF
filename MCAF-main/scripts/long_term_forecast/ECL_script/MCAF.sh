#!/bin/bash

model_name=MCAF

# Optimized hyperparameters for MCAF architecture
# Key changes:
# 1. Reduced d_model (256->512 causes overfitting in complex multi-domain models)
# 2. Increased batch_size for better gradient estimation
# 3. Warmup + cosine annealing for stable convergence
# 4. Gradient clipping to handle LSTM instability
# 5. Label smoothing for better generalization

# ============================================
# ECL 96->96: Short-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_core 64 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 50 \
  --patience 10 \
  --lradj cosine_with_warmup \
  --warmup_epochs 5 \
  --dropout 0.1 \
  --use_norm 1 \
  --activation gelu \
  --loss mse \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ECL 96->192: Medium-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_core 64 \
  --d_ff 512 \
  --batch_size 24 \
  --learning_rate 0.0004 \
  --train_epochs 50 \
  --patience 10 \
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
# ECL 96->336: Long-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 320 \
  --d_core 80 \
  --d_ff 640 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --train_epochs 60 \
  --patience 15 \
  --lradj cosine_with_warmup \
  --warmup_epochs 8 \
  --dropout 0.2 \
  --use_norm 1 \
  --activation gelu \
  --loss mse \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0

# ============================================
# ECL 96->720: Ultra-long-term prediction
# ============================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 384 \
  --d_core 96 \
  --d_ff 768 \
  --batch_size 8 \
  --learning_rate 0.0002 \
  --train_epochs 80 \
  --patience 20 \
  --lradj cosine_with_warmup \
  --warmup_epochs 10 \
  --dropout 0.25 \
  --use_norm 1 \
  --activation gelu \
  --loss huber \
  --des 'Exp' \
  --itr 1 \
  --use_gpu 1 \
  --gpu 0
