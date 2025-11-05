# src/train_lm.py
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import random
import numpy as np
from model import EncoderOnlyLM, Seq2SeqTransformer

# ---------------------------
# Dataset
# ---------------------------
class CharDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y

# ---------------------------
# Train loop
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/lm_base")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--ffn_hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_rope", action="store_true", help="Enable Rotary Position Embedding (RoPE)")
    ap.add_argument("--no_sdpa", action="store_true", help="Disable SDPA/FlashAttention acceleration")
    ap.add_argument("--use_decoder", action="store_true", help="Use decoder-only mode (experimental)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    text = open(args.data_path, encoding="utf-8").read()
    dataset = CharDataset(text, args.block_size)
    n = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n, len(dataset) - n])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_decoder:
        print("⚙️ Using decoder-only (Seq2SeqTransformer with tied src=tgt)")
        model = Seq2SeqTransformer(
            src_vocab=dataset.vocab_size, tgt_vocab=dataset.vocab_size,
            d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
            ffn_hidden=args.ffn_hidden, max_len=args.block_size + 1,
            dropout=args.dropout, use_rope=args.use_rope, use_sdpa=not args.no_sdpa
        )
    else:
        model = EncoderOnlyLM(
            vocab_size=dataset.vocab_size,
            d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
            ffn_hidden=args.ffn_hidden, max_len=args.block_size + 1,
            dropout=args.dropout, use_positional=True,
            use_rope=args.use_rope, use_sdpa=not args.no_sdpa
        )

    model = model.to(device)
    print(f"Vocab size: {dataset.vocab_size}, Parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)

    step = 0
    model.train()
    while step < args.max_steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if step % 50 == 0:
                with torch.no_grad():
                    model.eval()
                    val_losses = []
                    for xvb, yvb in val_loader:
                        xvb, yvb = xvb.to(device), yvb.to(device)
                        logits = model(xvb)
                        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yvb.view(-1))
                        val_losses.append(val_loss.item())
                    model.train()
                    vloss = sum(val_losses) / len(val_losses)
                    print(f"step {step:5d} | train {loss.item():.3f} | val {vloss:.3f} | ppl {math.exp(vloss):.1f}")
            step += 1
            if step >= args.max_steps:
                break

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final.pt"))
    print("✅ Training complete. Model saved at:", os.path.join(args.out_dir, "final.pt"))

if __name__ == "__main__":
    main()
