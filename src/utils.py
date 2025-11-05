import math, os, json, random
import torch
import numpy as np
from dataclasses import dataclass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lrs.append(base_lr * step / max(1, self.warmup_steps))
            else:
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                lrs.append(base_lr * 0.5 * (1 + math.cos(math.pi * progress)))
        return lrs

class JSONLLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a', encoding='utf-8')

    def log(self, **kwargs):
        self.f.write(json.dumps(kwargs) + '\n'); self.f.flush()

    def close(self):
        self.f.close()
