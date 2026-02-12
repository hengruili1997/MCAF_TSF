import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class AdaptiveTrendExtractor(nn.Module):
    """
    Method 3.1.1: Multi-scale Trend Extraction
    Optimization: Remove loops, use reshape for efficient Channel-Independent processing
    """
    def __init__(self, seq_len, scales=[3, 5, 7, 9]):
        super().__init__()
        self.scales = scales
        # Channel-Independent: treat B*N as new batch dimension
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=s, padding=s//2)
            for s in scales
        ])
        
        self.weight_net = nn.Sequential(
            nn.Linear(len(scales), 32),
            nn.ReLU(),
            nn.Linear(32, len(scales)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # Input x: [B, N, L]
        B, N, L = x.shape
        # Reshape to [B*N, 1, L] for batch processing
        x_flat = x.reshape(B*N, 1, L)
        
        features = []
        entropies = []
        
        for conv in self.convs:
            feat = conv(x_flat)  # [B*N, 1, L]
            features.append(feat)
            
            # Compute entropy for stability assessment
            prob = F.softmax(feat, dim=-1)  # Softmax over time
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)  # [B*N, 1]
            entropies.append(entropy)
            
        entropy_vec = torch.cat(entropies, dim=-1)  # [B*N, K]
        weights = self.weight_net(entropy_vec)  # [B*N, K]
        
        # Weighted fusion: weights [B*N, K] -> [B*N, K, 1, 1] for broadcasting
        weights = weights.unsqueeze(-1).unsqueeze(-1) 
        stacked_feats = torch.stack(features, dim=1)  # [B*N, K, 1, L]
        
        trend = (stacked_feats * weights).sum(dim=1)  # [B*N, 1, L]
        
        # Reshape back to [B, N, L]
        return trend.reshape(B, N, L)


class OptimizedDecompMAE(nn.Module):
    """
    Method 3.1.2: DecompMAE with optimizations
    - Batch STFT processing
    - Depthwise separable convolution
    - Hann window for spectral leakage reduction
    - Batch normalization for training stability
    """
    def __init__(self, seq_len, n_fft=64, mask_ratio=0.25):
        super().__init__()
        self.n_fft = n_fft
        self.mask_ratio = mask_ratio
        self.hop_length = n_fft // 4
        
        # Depthwise separable convolution for parameter efficiency
        self.recon_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(16),  # Stabilize training
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),  # Depthwise
            nn.Conv2d(16, 1, kernel_size=1)  # Pointwise
        )

    def forward(self, x_det):
        # x_det: [B, N, L]
        B, N, L = x_det.shape
        x_flat = x_det.reshape(B*N, L)  # [B*N, L]
        
        # Use Hann window to reduce spectral leakage
        window = torch.hann_window(self.n_fft, device=x_flat.device)
        
        # Batch STFT processing
        stft = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
                         window=window, return_complex=True, normalized=True, onesided=True)
        # stft: [B*N, F, T]
        
        amp = torch.abs(stft)
        phase = torch.angle(stft)
        
        # Dynamic masking strategy
        if self.training:
            mask = torch.rand_like(amp) > self.mask_ratio
            amp_masked = amp * mask
        else:
            amp_masked = amp
            
        # Reconstruction network: [B*N, 1, F, T]
        amp_recon = self.recon_net(amp_masked.unsqueeze(1)).squeeze(1)
        
        # Seasonal component (reconstructed harmonic)
        seasonal_spec = torch.polar(amp_recon, phase)
        seasonal_flat = torch.istft(seasonal_spec, n_fft=self.n_fft,
                                    hop_length=self.hop_length, length=L,
                                    window=window, normalized=True)
        seasonal = seasonal_flat.reshape(B, N, L)
        
        # Residual component (noise) - keep in time-frequency domain
        residual_amp = amp - amp_recon
        residual_spec = torch.polar(residual_amp, phase)  # [B*N, F, T] Complex
        
        # Reshape residual to [B, N, F, T]
        F_dim, T_dim = residual_spec.shape[1], residual_spec.shape[2]
        residual_spec = residual_spec.reshape(B, N, F_dim, T_dim)
        
        return seasonal, residual_spec


class TrendLSTM(nn.Module):
    """
    Method 3.2.1: Time Domain Trend Modeling
    Adaptive basis function + LSTM predictor
    """
    def __init__(self, seq_len, pred_len, hidden_dim=64, num_basis=8):
        super().__init__()
        self.pred_len = pred_len
        self.num_basis = num_basis
        
        # Adaptive basis function generator
        self.basis_gen = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_basis),
            nn.Softmax(dim=-1)
        )
        
        # LSTM predictor: input_dim = 1 (trend value) + num_basis (global context)
        self.lstm = nn.LSTM(input_size=1 + num_basis, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, trend):
        # trend: [B, N, L]
        B, N, L = trend.shape
        trend_flat = trend.reshape(B*N, L, 1)  # [Batch', Seq, Feat]
        
        # Global statistics -> basis weights
        global_stat = trend_flat.mean(dim=1)  # [B*N, 1]
        basis_weights = self.basis_gen(global_stat)  # [B*N, num_basis]
        
        # Expand basis weights to each time step
        # basis_weights: [B*N, 1, num_basis] -> [B*N, L, num_basis]
        basis_weights_expanded = basis_weights.unsqueeze(1).repeat(1, L, 1)
        
        # Concatenate: [B*N, L, 1] + [B*N, L, num_basis] -> [B*N, L, 1+num_basis]
        lstm_input = torch.cat([trend_flat, basis_weights_expanded], dim=2)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_input)  # [B*N, L, Hidden]
        
        # Use last time step for prediction
        pred = self.proj(lstm_out[:, -1, :])  # [B*N, pred_len]
        
        return pred.reshape(B, N, self.pred_len)


class SeasonalFreqEnhancer(nn.Module):
    """
    Method 3.2.2: Frequency Domain Seasonal Modeling
    Top-K sparse enhancement with vectorized processing
    """
    def __init__(self, seq_len, pred_len, topk_ratio=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.topk_ratio = topk_ratio
        
        # Amplitude enhancement network
        self.amp_enhancer = nn.Sequential(
            nn.Linear(1, 16), 
            nn.ReLU(), 
            nn.Linear(16, 1)
        )

    def forward(self, seasonal):
        # seasonal: [B, N, L]
        B, N, L = seasonal.shape
        x_flat = seasonal.reshape(B*N, L)
        
        # Real FFT
        fft = torch.fft.rfft(x_flat, dim=-1)
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Top-K selection
        k = max(1, int(amp.size(-1) * self.topk_ratio))
        topk_vals, topk_idx = torch.topk(amp, k, dim=-1)
        
        # Enhance Top-K amplitudes
        # [B*N, k, 1] -> MLP -> [B*N, k, 1]
        enhanced_vals = self.amp_enhancer(topk_vals.unsqueeze(-1)).squeeze(-1)
        
        # Reconstruct enhanced spectrum
        amp_new = torch.zeros_like(amp)
        amp_new.scatter_(1, topk_idx, enhanced_vals)
        
        # Inverse transform (generate pred_len length)
        fft_enhanced = torch.polar(amp_new, phase)
        pred = torch.fft.irfft(fft_enhanced, n=self.pred_len, dim=-1)
        
        return pred.reshape(B, N, self.pred_len)


class ResidualDictLearning(nn.Module):
    """
    Method 3.2.3: Residual Modeling in Time-Frequency Domain
    Simplified with CNN-based mapping for efficiency
    """
    def __init__(self, time_len, freq_len, pred_len, dict_size=64, num_iter=2):
        super().__init__()
        self.num_iter = num_iter
        self.pred_len = pred_len
        self.lambda_sparse = 0.1
        
        # CNN mapper: time-frequency residual -> future prediction
        self.mapper = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Flatten(),
        )
        # Lazy linear for flexible input dimensions
        self.output_proj = nn.LazyLinear(pred_len * 2)  # *2 for real/imag

    def forward(self, residual_spec):
        # residual_spec: [B, N, F, T] (Complex)
        B, N, F, T = residual_spec.shape
        
        # Use amplitude as input for CNN
        res_amp = torch.abs(residual_spec).reshape(B*N, 1, F, T)
        
        # Map to future residual
        features = self.mapper(res_amp)  # [B*N, flattened]
        pred_flat = self.output_proj(features)  # [B*N, pred_len * 2]
        
        # Reshape to complex [B, N, pred_len]
        pred_flat = pred_flat.reshape(B, N, self.pred_len, 2)
        pred_complex = torch.view_as_complex(pred_flat.contiguous())  # [B, N, pred_len]
        
        return pred_complex


class ComplexFusion(nn.Module):
    """
    Method 3.3: Complex-valued Fusion with Multi-head Attention
    Maps all components to unified complex representation space
    """
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len
        
        # Complex projection layer
        self.complex_proj = nn.Linear(pred_len, pred_len, dtype=torch.cfloat)
        
        # Attention scoring network
        self.attn_score = nn.Sequential(
            nn.Linear(pred_len, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, trend, seasonal, residual):
        # All inputs: [B, N, pred_len]
        # Trend/Seasonal are real, Residual is complex
        
        # Project to unified complex space
        trend_c = self.complex_proj(trend.to(torch.cfloat))
        seasonal_c = self.complex_proj(seasonal.to(torch.cfloat))
        residual_c = residual  # Already complex from ResidualDictLearning
        
        # Stack: [B, N, 3, pred_len]
        stack = torch.stack([trend_c, seasonal_c, residual_c], dim=2)
        
        # Attention based on amplitude
        amp = torch.abs(stack)  # [B, N, 3, pred_len]
        score = self.attn_score(amp).squeeze(-1)  # [B, N, 3]
        weights = F.softmax(score, dim=-1).unsqueeze(-1)  # [B, N, 3, 1]
        
        # Weighted fusion
        fused = (stack * weights).sum(dim=2)  # [B, N, pred_len]
        
        return fused.real


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with STAR attention mechanism
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Stacked Encoder Layers
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DataEmbedding_inverted(nn.Module):
    """
    Inverted embedding: treat each variable as a token
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [B, N, L]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class STAR(nn.Module):
    """
    STar Aggregate-Redistribute Module
    Channel-independent cross-variable information sharing
    """
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # FFN for feature extraction
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # Stochastic pooling with numerical stability
        if self.training:
            combined_mean_stable = combined_mean - combined_mean.max(dim=1, keepdim=True).values
            ratio = F.softmax(combined_mean_stable, dim=1)
            ratio = ratio.permute(0, 2, 1).reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            combined_mean_stable = combined_mean - combined_mean.max(dim=1, keepdim=True).values
            weight = F.softmax(combined_mean_stable, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # MLP fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        return combined_mean_cat, None


class Model(nn.Module):
    """
    MCAF: Multi-domain Collaborative Analytics Framework
    Main model integrating all components
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        
        # Step 1: Signal Decomposition
        self.trend_extractor = AdaptiveTrendExtractor(configs.seq_len)
        self.decomp_mae = OptimizedDecompMAE(configs.seq_len)
        
        # Step 2: Categorical Processing
        self.trend_model = TrendLSTM(configs.seq_len, configs.pred_len)
        self.seasonal_model = SeasonalFreqEnhancer(configs.seq_len, configs.pred_len)
        # Frequency dimension depends on STFT parameters (n_fft=64 -> F=33)
        self.residual_model = ResidualDictLearning(configs.seq_len, 33, configs.pred_len)
        
        # Step 3: Feature Aggregation
        self.fusion = ComplexFusion(configs.pred_len)
        
        # Optional: STAR enhancement
        self.enc_embedding = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                STAR(configs.d_model, configs.d_core),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for _ in range(configs.e_layers)
        ])
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization (RevIN)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        # Convert to [B, N, L] format
        x_norm = x_enc.permute(0, 2, 1)  # [B, L, N] -> [B, N, L]
        
        # --- Step 1: Decompose ---
        trend_part = self.trend_extractor(x_norm)
        x_det = x_norm - trend_part
        seasonal_part, residual_spec = self.decomp_mae(x_det)
        
        # --- Step 2: Modeling (Forecast) ---
        trend_pred = self.trend_model(trend_part)       # [B, N, pred_len]
        seasonal_pred = self.seasonal_model(seasonal_part)  # [B, N, pred_len]
        residual_pred = self.residual_model(residual_spec)  # [B, N, pred_len] (Complex)
        
        # --- Step 3: Fusion ---
        final_pred = self.fusion(trend_pred, seasonal_pred, residual_pred)  # [B, N, pred_len]
        
        # Optional: STAR enhancement
        enc_out = self.enc_embedding(final_pred.transpose(1, 2), None)  # [B, N, d_model]
        enc_out, _ = self.encoder(enc_out)
        dec_out = self.projection(enc_out).permute(0, 2, 1)  # [B, pred_len, N]
        
        # Denormalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, N]


