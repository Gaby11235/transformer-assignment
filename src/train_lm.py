import os, math, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import set_seed, CosineWithWarmup, JSONLLogger
from model import EncoderOnlyLM, make_causal_mask

class CharDataset(Dataset):
    def __init__(self, path, block_size=256, split='train'):
        text = open(path, 'r', encoding='utf-8').read()
        n = len(text)
        n_train = int(0.9 * n)
        n_val   = int(0.05 * n)
        if split == 'train':
            text = text[:n_train]
        elif split == 'val':
            text = text[n_train:n_train+n_val]
        else:
            text = text[n_train+n_val:]

        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.chars)
        self.data = [self.stoi[c] for c in text]
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, default='data/tiny_shakespeare_sample.txt')
    ap.add_argument('--out_dir', type=str, default='results/lm_base')
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--block_size', type=int, default=256)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_head', type=int, default=4)
    ap.add_argument('--n_layer', type=int, default=4)
    ap.add_argument('--ffn_hidden', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup_steps', type=int, default=200)
    ap.add_argument('--max_steps', type=int, default=2500)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--no_positional_encoding', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = CharDataset(args.data_path, block_size=args.block_size, split='train')
    val_ds   = CharDataset(args.data_path, block_size=args.block_size, split='val')

    model = EncoderOnlyLM(
        vocab_size=train_ds.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_hidden=args.ffn_hidden,
        max_len=args.block_size+1,
        dropout=0.1,
        use_positional=(not args.no_positional_encoding)
    ).to(device)

    print(f"Vocab size: {train_ds.vocab_size}, Parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = CosineWithWarmup(opt, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(args.out_dir, 'train_log.jsonl'))

    step = 0
    best_val = float('inf')
    ce = nn.CrossEntropyLoss()

    while step < args.max_steps:
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # causal mask: [B,1,T,T]
            T = xb.size(1)
            causal = make_causal_mask(T, device=device)
            logits = model(xb, src_mask=causal)
            loss = ce(logits.view(-1, logits.size(-1)), yb.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    vloss_total, vcount = 0.0, 0
                    for xvb, yvb in val_loader:
                        xvb, yvb = xvb.to(device), yvb.to(device)
                        T = xvb.size(1)
                        causal_v = make_causal_mask(T, device=device)
                        v_logits = model(xvb, src_mask=causal_v)
                        v_loss = ce(v_logits.view(-1, v_logits.size(-1)), yvb.view(-1))
                        vloss_total += v_loss.item()
                        vcount += 1
                vloss = vloss_total / max(1, vcount)
                ppl = math.exp(min(20, vloss))
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:5d} | train {loss.item():.3f} | val {vloss:.3f} | ppl {ppl:.1f} | lr {lr_now:.6f}")

                logger.log(step=step, train_loss=float(loss.item()), val_loss=float(vloss), ppl=float(ppl), lr=float(lr_now))

                if vloss < best_val:
                    best_val = vloss
                    save_checkpoint(model, opt, step, os.path.join(args.out_dir, 'best.pt'))

            step += 1
            if step >= args.max_steps:
                break

    # save final
    save_checkpoint(model, opt, step, os.path.join(args.out_dir, 'final.pt'))
    logger.close()

if __name__ == '__main__':
    main()
