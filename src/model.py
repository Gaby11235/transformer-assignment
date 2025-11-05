import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- Positional Encoding -----------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ----------------- Attention primitives -----------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    # q: [B, h, T_q, d], k: [B, h, T_k, d], v: [B, h, T_k, d]
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,h,T_q,T_k]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=q.requires_grad)
    out = torch.matmul(attn, v)  # [B,h,T_q,d]
    return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_dropout=0.0, proj_dropout=0.0):
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

    def forward(self, x_q, x_kv, mask=None):
        B, Tq, C = x_q.shape
        Tk = x_kv.shape[1]
        q = self.w_q(x_q)  # [B,Tq,C]
        k = self.w_k(x_kv) # [B,Tk,C]
        v = self.w_v(x_kv) # [B,Tk,C]

        # split heads
        q = q.view(B, Tq, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tq,d]
        k = k.view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tk,d]
        v = v.view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # [B,h,Tk,d]

        out, attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=self.attn_dropout)  # [B,h,Tq,d]
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)  # [B,Tq,C]
        out = self.out_proj(out)
        if self.proj_dropout > 0:
            out = F.dropout(out, p=self.proj_dropout, training=self.training)
        return out, attn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout, proj_dropout=resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x, src_mask=None):
        # Self-attention
        y, _ = self.mha(self.ln1(x), self.ln1(x), mask=src_mask)
        x = x + y
        # FFN
        y = self.ff(self.ln2(x))
        x = x + y
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout, proj_dropout=resid_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, attn_dropout=attn_dropout, proj_dropout=resid_dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x, enc, tgt_mask=None, mem_mask=None):
        y, _ = self.self_attn(self.ln1(x), self.ln1(x), mask=tgt_mask)  # causal self-attn
        x = x + y
        y, _ = self.cross_attn(self.ln2(x), enc, mask=mem_mask)          # encoder-decoder attn
        x = x + y
        y = self.ff(self.ln3(x))
        x = x + y
        return x

# ----------------- Models -----------------
class EncoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, ffn_hidden, max_len, dropout=0.1, use_positional=True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len) if use_positional else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_head, ffn_hidden, attn_dropout=dropout, resid_dropout=dropout)
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
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, n_layer, ffn_hidden, max_len, dropout=0.1):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_head, ffn_hidden, attn_dropout=dropout, resid_dropout=dropout)
            for _ in range(n_layer)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_head, ffn_hidden, attn_dropout=dropout, resid_dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_e = nn.LayerNorm(d_model)
        self.ln_d = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, tgt_vocab, bias=False)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        # src,tgt: [B,T]
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

# ----------------- Mask helpers -----------------
def make_padding_mask(x, pad_id=0):
    # x: [B,T], returns [B,1,1,T] broadcastable
    mask = (x != pad_id).unsqueeze(1).unsqueeze(2)
    return mask  # bool

def make_causal_mask(sz: int, device):
    return torch.tril(torch.ones((sz, sz), device=device)).bool().unsqueeze(0).unsqueeze(1)
