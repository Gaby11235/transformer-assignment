import os, math, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import set_seed, CosineWithWarmup, JSONLLogger
from model import Seq2SeqTransformer, make_causal_mask, make_padding_mask

PAD = 0
BOS = 1
EOS = 2

class CopyTaskDataset(Dataset):
    def __init__(self, vocab_size=40, max_len=40, size=10000, split='train', seed=123):
        rnd = random.Random(seed if split=='train' else seed+1 if split=='val' else seed+2)
        self.samples = []
        for _ in range(size):
            L = rnd.randint(5, max_len)
            seq = [rnd.randint(3, vocab_size-1) for _ in range(L)]
            self.samples.append(seq)

        self.vocab_size = vocab_size
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src = self.samples[idx]
        tgt = src[:]  # copy
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_batch(batch):
    # pad and add BOS/EOS
    max_src = max(len(s) for s,_ in batch)
    max_tgt = max(len(t) for _,t in batch) + 1  # +1 for BOS (decoder input length)
    srcs, tgts_in, tgts_out = [], [], []
    for src, tgt in batch:
        s = torch.full((max_src,), PAD, dtype=torch.long)
        s[:len(src)] = src
        srcs.append(s)

        t_in = torch.full((max_tgt,), PAD, dtype=torch.long)
        t_out= torch.full((max_tgt,), PAD, dtype=torch.long)
        t_in[0] = BOS
        t_in[1:len(tgt)+1] = tgt
        t_out[:len(tgt)] = tgt
        t_out[len(tgt)] = EOS
        tgts_in.append(t_in)
        tgts_out.append(t_out)

    return torch.stack(srcs), torch.stack(tgts_in), torch.stack(tgts_out)

def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_size', type=int, default=40)
    ap.add_argument('--max_len', type=int, default=40)
    ap.add_argument('--size', type=int, default=10000)
    ap.add_argument('--out_dir', type=str, default='results/mt_copy')
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_head', type=int, default=4)
    ap.add_argument('--n_layer', type=int, default=3)
    ap.add_argument('--ffn_hidden', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup_steps', type=int, default=200)
    ap.add_argument('--max_steps', type=int, default=3000)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = CopyTaskDataset(args.vocab_size, args.max_len, size=args.size, split='train')
    val_ds   = CopyTaskDataset(args.vocab_size, args.max_len, size=args.size//10, split='val')

    model = Seq2SeqTransformer(
        src_vocab=args.vocab_size,
        tgt_vocab=args.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_hidden=args.ffn_hidden,
        max_len=args.max_len+2,
        dropout=0.1
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = CosineWithWarmup(opt, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_batch)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(args.out_dir, 'train_log.jsonl'))

    step = 0
    best_val = float('inf')
    ce = nn.CrossEntropyLoss(ignore_index=PAD)

    while step < args.max_steps:
        model.train()
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask = make_padding_mask(src, pad_id=PAD)
            tgt_mask = make_padding_mask(tgt_in, pad_id=PAD) & make_causal_mask(tgt_in.size(1), device)
            mem_mask = make_padding_mask(src, pad_id=PAD)

            logits = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask, mem_mask=mem_mask)
            loss = ce(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    vloss_total, vcount = 0.0, 0
                    for vsrc, vtgt_in, vtgt_out in val_loader:
                        vsrc, vtgt_in, vtgt_out = vsrc.to(device), vtgt_in.to(device), vtgt_out.to(device)
                        v_src_mask = make_padding_mask(vsrc, pad_id=PAD)
                        v_tgt_mask = make_padding_mask(vtgt_in, pad_id=PAD) & make_causal_mask(vtgt_in.size(1), device)
                        v_mem_mask = make_padding_mask(vsrc, pad_id=PAD)
                        v_logits = model(vsrc, vtgt_in, src_mask=v_src_mask, tgt_mask=v_tgt_mask, mem_mask=v_mem_mask)
                        v_loss = ce(v_logits.view(-1, v_logits.size(-1)), vtgt_out.view(-1))
                        vloss_total += v_loss.item()
                        vcount += 1
                vloss = vloss_total / max(1, vcount)
                lr_now = sched.get_last_lr()[0]
                print(f"step {step:5d} | train {loss.item():.3f} | val {vloss:.3f} | lr {lr_now:.6f}")
                logger.log(step=step, train_loss=float(loss.item()), val_loss=float(vloss), lr=float(lr_now))

                if vloss < best_val:
                    best_val = vloss
                    save_checkpoint(model, opt, step, os.path.join(args.out_dir, 'best.pt'))

            step += 1
            if step >= args.max_steps:
                break

    save_checkpoint(model, opt, step, os.path.join(args.out_dir, 'final.pt'))
    logger.close()

if __name__ == '__main__':
    main()
