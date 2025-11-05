# src/train_mt.py
import math
import os
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from model import Seq2SeqTransformer

# --------------------------
# Toy Copy Task Dataset
# --------------------------
class CopyTaskDataset(Dataset):
    def __init__(self, vocab_size=40, max_len=40, n_samples=5000):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.samples = []
        for _ in range(n_samples):
            L = torch.randint(5, max_len, (1,)).item()
            seq = torch.randint(2, vocab_size, (L,))
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src = self.samples[idx]
        tgt_in = torch.cat([torch.tensor([1]), src])[:self.max_len]  # <bos>=1
        tgt_out = torch.cat([src, torch.tensor([2])])[:self.max_len]  # <eos>=2
        return src, tgt_in, tgt_out


def collate_fn(batch):
    srcs, tgts_in, tgts_out = zip(*batch)
    pad = 0
    max_len = max(len(s) for s in srcs)

    def pad_seq(seqs):
        return torch.stack([F.pad(s, (0, max_len - len(s)), value=pad) for s in seqs])

    return pad_seq(srcs), pad_seq(tgts_in), pad_seq(tgts_out)


# --------------------------
# Train loop
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--ffn_hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out_dir", type=str, default="results/mt_copy")
    ap.add_argument("--use_rope", action="store_true", help="Enable Rotary Positional Encoding")
    ap.add_argument("--no_sdpa", action="store_true", help="Disable SDPA/FlashAttention")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = CopyTaskDataset(vocab_size=args.vocab_size, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Seq2SeqTransformer(
        src_vocab=args.vocab_size,
        tgt_vocab=args.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_hidden=args.ffn_hidden,
        max_len=args.max_len,
        dropout=args.dropout,
        use_rope=args.use_rope,
        use_sdpa=not args.no_sdpa
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M on {device}")

    step = 0
    loss_history = []
    progress_bar = tqdm(total=args.max_steps, desc="Training", ncols=100)

    while step < args.max_steps:
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            opt.zero_grad()
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # 记录 loss
            loss_history.append(loss.item())
            if step % 50 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})
            step += 1
            progress_bar.update(1)

            if step >= args.max_steps:
                break

    progress_bar.close()
    print("✅ Training complete.")

    # 保存模型
    model_path = os.path.join(args.out_dir, "final.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # 保存 loss 日志
    csv_path = os.path.join(args.out_dir, "loss_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, l in enumerate(loss_history):
            writer.writerow([i + 1, l])
    print(f"Loss log saved to {csv_path}")

    # 绘制 loss 曲线
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, label="Training Loss", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Copy Task)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt_path = os.path.join(args.out_dir, "loss_curve.png")
    plt.savefig(plt_path)
    print(f"Loss curve saved to {plt_path}")


if __name__ == "__main__":
    main()
