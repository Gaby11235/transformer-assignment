# Mini Transformer (From Scratch)

A compact PyTorch implementation of an encoder-only Transformer for character-level language modeling (Tiny Shakespeare),
plus an optional encoder–decoder for toy seq2seq tasks. Includes training utilities (AdamW, cosine LR, warmup, gradient
clipping, checkpointing, curve plotting) and a LaTeX report template.

> **Reproducibility:** Exact commands with seeds are provided below.

## Environment

```bash
conda create -n transformer python=3.10 -y
conda activate transformer
pip install -r requirements.txt
```

## Data

The repo includes a small placeholder text at `data/tiny_shakespeare_sample.txt` so that training runs out of the box.
For better results, replace it with the full Tiny Shakespeare file (public domain) named the same, or pass `--data_path`.
You can also try WikiText-2 or PTB after adapting the `CharDataset` to a word-level tokenizer.

## Train (Encoder-only LM)

```bash
python src/train_lm.py \  --data_path data/tiny_shakespeare_sample.txt \  --out_dir results/lm_base \  --seed 3407 \  --batch_size 64 \  --block_size 256 \  --d_model 256 \  --n_head 4 \  --n_layer 4 \  --ffn_hidden 1024 \  --lr 3e-4 \  --warmup_steps 200 \  --max_steps 2500
```

## Train (Seq2Seq toy: copy task)

```bash
python src/train_mt.py \  --task copy --vocab_size 40 --max_len 40 \  --out_dir results/mt_copy \  --seed 3407 \  --batch_size 64 \  --d_model 256 \  --n_head 4 \  --n_layer 3 \  --ffn_hidden 1024 \  --lr 3e-4 \  --warmup_steps 200 \  --max_steps 3000
```

## Plotting training curves

Both trainers write `train_log.jsonl`. To plot, just run:
```bash
python src/plot_curves.py --log results/lm_base/train_log.jsonl --out results/lm_base/curve.png
```

## Repo structure

```
.
├── configs/
│   └── base.yaml
├── data/
│   └── tiny_shakespeare_sample.txt
├── report/
│   ├── report.tex
│   └── references.bib
├── results/           # training outputs go here (curves, checkpoints, tables)
├── scripts/
│   └── run.sh
├── src/
│   ├── model.py
│   ├── train_lm.py
│   ├── train_mt.py
│   ├── plot_curves.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Hardware

- Designed for CPU or a single small GPU (e.g., 4–8 GB). For Tiny Shakespeare @ char-level, training converges in minutes on a notebook GPU.
- Mixed precision is not enabled by default (kept simple), but you can wrap with `torch.autocast` easily.

## Notes

- Implements: multi-head self-attention, position-wise FFN, residual+LayerNorm, sinusoidal positional encoding, padding & causal masks.
- Tricks: AdamW, cosine LR with warmup, gradient clipping, checkpoint save/load, deterministic seeding, parameter count printing.
- Ablations: disable positional encoding (`--no_positional_encoding`), vary number of heads/layers, or switch to the seq2seq toy task.

## License

MIT
