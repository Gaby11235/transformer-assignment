# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Positional Encodings
# =========================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Absolute sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T, :]

# =========================================================
# Relative Positional Encoding: RoPE
# =========================================================

def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    Rotary Positional Embedding (RoPE) applied to last dim pairs.
    x: [B, H, T, D], where D must be even.
    Returns x_rot with the same shape.
    """
    B, H, T, D = x.shape
    assert D % 2 == 0, "RoPE requires head_dim to be even."
    half = D // 2

    # Frequencies
    freq_seq = torch.arange(half, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (freq_seq / half))  # [half]
    pos = torch.arange(T, device=x.device, dtype=x.dtype)  # [T]
    sinusoid = torch.einsum("t,f->tf", pos, inv_freq)      # [T, half]

    sin = sinusoid.sin()[None, None, :, :]  # [1,1,T,half]
    cos = sinusoid.cos()[None, None, :, :]  # [1,1,T,half]

    x1, x2 = x[..., :half], x[..., half:]
    # (a + ib) * (cos + i sin) = (a cos - b sin) + i(a sin + b cos)
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

# =========================================================
# Attention Core (SDPA fallback)
# =========================================================

def sdpa_available() -> bool:
    return hasattr(F, "scaled_dot_product_attention")

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, is_causal=False, use_sdpa=True):
    """
    q, k, v: [B, H, T_q/k, D]
    mask: optional boolean mask broadcastable to [B, H, T_q, T_k]
          True = keep, False = mask out.
    If use_sdpa and PyTorch supports it, route to F.scaled_dot_product_attention
    with an additive mask; otherwise fallback to manual computation.
    """
    if use_sdpa and sdpa_available():
        attn_mask = None
        if mask is not None:
            # Build additive mask: 0 for keep, -inf for masked
            # SDPA expects float mask being added to attention scores.
            attn_mask = (~mask).to(q.dtype) * float("-inf")  # False->-inf, True->0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,              # [B,H,Tq,Tk] or broadcastable
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=is_causal
        )
        # out: [B,H,Tq,D]
        # For completeness, also return attn probs only for API compatibility (None here).
        return out, None

    # --------- Fallback to manual attention ----------
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,Tq,Tk]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    if is_causal:
        Tq, Tk = scores.size(-2), scores.size(-1)
        causal = torch.tril(torch.ones((Tq, Tk), device=scores.device)).bool()
        scores = scores.masked_fill(~causal, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=q.requires_grad)
    out = torch.matmul(attn, v)  # [B,H,Tq,D]
    return out, attn

# =========================================================
# Multi-Head Attention
# =========================================================

class MultiHeadAttention(nn.Module):
    """
    Supports self-attention and cross-attention.
    Options:
      - use_rope: apply RoPE to (q,k)
      - use_sdpa: use PyTorch SDPA/FlashAttention when available, else fallback
    """
    def __init__(self, d_model, n_head, attn_dropout=0.0, proj_dropout=0.0,
                 use_rope: bool = False, use_sdpa: bool = True):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.use_rope = use_rope
        self.use_sdpa = use_sdpa

    def forward(self, x_q, x_kv, mask=None, is_causal: bool = False):
        """
        x_q:  [B, Tq, C]
        x_kv: [B, Tk, C]
        mask: (optional) boolean mask broadcastable to [B, 1 or H, Tq, Tk]; True=keep, False=mask
        is_causal: set True for decoder self-attention
        """
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]

        q = self.w_q(x_q)  # [B,Tq,C]
        k = self.w_k(x_kv) # [B,Tk,C]
        v = self.w_v(x_kv) # [B,Tk,C]

        # reshape to multihead
        q = q.view(B, Tq, self.n_head, self.d_head).transpose(1, 2)  # [B,H,Tq,D]
        k = k.view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,H,Tk,D]
        v = v.view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,H,Tk,D]

        # RoPE on q,k (if enabled)
        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        # Ensure mask has shape [B,H,Tq,Tk] if provided
        if mask is not None:
            if mask.dim() == 4:
                # [B,1,Tq,Tk] -> [B,H,Tq,Tk]
                if mask.size(1) == 1 and self.n_head > 1:
                    mask = mask.expand(B, self.n_head, mask.size(2), mask.size(3))
            elif mask.dim() == 3:
                # [B,Tq,Tk] -> [B,H,Tq,Tk]
                mask = mask.unsqueeze(1).expand(B, self.n_head, mask.size(1), mask.size(2))
            else:
                # [B,1,1,Tk] or similar; try to broadcast later
                pass

        out, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.attn_dropout, is_causal=is_causal, use_sdpa=self.use_sdpa
        )
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # [B,Tq,C]
        out = self.out_proj(out)
        if self.proj_dropout > 0:
            out = F.dropout(out, p=self.proj_dropout, training=self.training)
        return out

# =========================================================
# Encoder / Decoder Blocks (Pre-LN)
# =========================================================

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, attn_dropout=0.0, resid_dropout=0.0,
                 use_rope: bool = False, use_sdpa: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout,
                                      proj_dropout=resid_dropout, use_rope=use_rope, use_sdpa=use_sdpa)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x, src_mask=None):
        y = self.mha(self.ln1(x), self.ln1(x), mask=src_mask, is_causal=False)
        x = x + y
        y = self.ff(self.ln2(x))
        x = x + y
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, attn_dropout=0.0, resid_dropout=0.0,
                 use_rope: bool = False, use_sdpa: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout,
                                            proj_dropout=resid_dropout, use_rope=use_rope, use_sdpa=use_sdpa)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout,
                                             proj_dropout=resid_dropout, use_rope=use_rope, use_sdpa=use_sdpa)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x, enc, tgt_mask=None, mem_mask=None):
        # Decoder self-attention (causal)
        y = self.self_attn(self.ln1(x), self.ln1(x), mask=tgt_mask, is_causal=True)
        x = x + y
        # Cross attention (non-causal)
        y = self.cross_attn(self.ln2(x), enc, mask=mem_mask, is_causal=False)
        x = x + y
        y = self.ff(self.ln3(x))
        x = x + y
        return x

# =========================================================
# Models
# =========================================================

class EncoderOnlyLM(nn.Module):
    """Encoder-only Transformer for (character-level) language modeling."""
    def __init__(self, vocab_size, d_model, n_head, n_layer, ffn_hidden, max_len,
                 dropout=0.1, use_positional=True, use_rope: bool = False, use_sdpa: bool = True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len) if use_positional else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_head, ffn_hidden,
                                    attn_dropout=dropout, resid_dropout=dropout,
                                    use_rope=use_rope, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, src_mask=None):
        # idx: [B,T]
        x = self.tok_emb(idx)  # [B,T,C]
        x = self.pos_enc(x)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, src_mask=src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class Seq2SeqTransformer(nn.Module):
    """Standard Transformer Encoder–Decoder."""
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, n_layer, ffn_hidden, max_len,
                 dropout=0.1, use_rope: bool = False, use_sdpa: bool = True):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_head, ffn_hidden,
                                    attn_dropout=dropout, resid_dropout=dropout,
                                    use_rope=use_rope, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_head, ffn_hidden,
                                    attn_dropout=dropout, resid_dropout=dropout,
                                    use_rope=use_rope, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])
        self.ln_e = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tgt_vocab, bias=False)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        # src, tgt: [B,T]
        enc = self.drop(self.pos_enc(self.src_emb(src)))
        for blk in self.encoder:
            enc = blk(enc, src_mask=src_mask)
        enc = self.ln_e(enc)

        dec = self.drop(self.pos_enc(self.tgt_emb(tgt)))
        for blk in self.decoder:
            dec = blk(dec, enc, tgt_mask=tgt_mask, mem_mask=mem_mask)
        dec = self.ln_d(dec)
        logits = self.head(dec)
        return logits

# =========================================================
# Mask helpers
# =========================================================

def make_padding_mask(x: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    x: [B,T] token ids
    returns: [B,1,1,T] bool mask (True=keep, False=mask out)
    """
    return (x != pad_id).unsqueeze(1).unsqueeze(2)

def make_causal_mask(T_q: int, T_k: int = None, device=None) -> torch.Tensor:
    """
    Build a lower-triangular causal mask.
    returns: [1,1,T_q,T_k] bool mask
    """
    if T_k is None:
        T_k = T_q
    m = torch.tril(torch.ones((T_q, T_k), device=device)).bool()
    return m.unsqueeze(0).unsqueeze(0)
