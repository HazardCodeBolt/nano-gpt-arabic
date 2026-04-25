"""
NanoGPT training script for Arabic Text Generation.
Based on Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT).

Usage:
    # Prepare data first (run once):
    python data_prepare.py

    # Train with defaults:
    python train.py

    # Override any config value:
    python train.py --n_layer=4 --n_embd=256 --max_iters=3000

    # Resume from checkpoint:
    python train.py --init_from=resume
"""

import os
import sys
import math
import time
import pickle
import argparse
import numpy as np
import torch

from model import GPT, GPTConfig

# ── Argument parsing (overrides defaults below) ───────────────────────────
parser = argparse.ArgumentParser(description='Train NanoGPT on Arabic text')

# I/O
parser.add_argument('--out_dir',           default='out',      type=str)
parser.add_argument('--data_dir',          default='data',     type=str)
parser.add_argument('--init_from',         default='scratch',  type=str,
                    choices=['scratch', 'resume'],
                    help='scratch = new model, resume = continue from out_dir/ckpt.pt')

# Logging / checkpointing
parser.add_argument('--eval_interval',     default=250,   type=int)
parser.add_argument('--log_interval',      default=50,    type=int)
parser.add_argument('--eval_iters',        default=100,   type=int)
parser.add_argument('--always_save_ckpt',  default=True,  type=lambda x: x.lower() != 'false')

# Data
parser.add_argument('--block_size',        default=256,   type=int)
parser.add_argument('--batch_size',        default=64,    type=int)

# Model
parser.add_argument('--n_layer',           default=6,     type=int)
parser.add_argument('--n_head',            default=6,     type=int)
parser.add_argument('--n_embd',            default=384,   type=int)
parser.add_argument('--dropout',           default=0.1,   type=float)
parser.add_argument('--bias',              default=False, type=lambda x: x.lower() == 'true')

# Optimiser
parser.add_argument('--learning_rate',     default=3e-4,  type=float)
parser.add_argument('--max_iters',         default=5000,  type=int)
parser.add_argument('--weight_decay',      default=0.1,   type=float)
parser.add_argument('--beta1',             default=0.9,   type=float)
parser.add_argument('--beta2',             default=0.95,  type=float)
parser.add_argument('--grad_clip',         default=1.0,   type=float)

# LR schedule
parser.add_argument('--decay_lr',          default=True,  type=lambda x: x.lower() != 'false')
parser.add_argument('--warmup_iters',      default=200,   type=int)
parser.add_argument('--min_lr',            default=3e-5,  type=float)

# System
parser.add_argument('--device',            default='',    type=str,
                    help="'' = auto-detect, or 'cpu', 'cuda', 'mps'")
parser.add_argument('--compile',           default=False, type=lambda x: x.lower() == 'true',
                    help='Use torch.compile() — requires PyTorch 2.0+')

args = parser.parse_args()

# ── Device setup ─────────────────────────────────────────────────────────
if args.device:
    device = args.device
elif torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

device_type = 'cuda' if 'cuda' in device else 'cpu'
print(f"Device: {device}")

torch.manual_seed(1337)
if device_type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

dtype = (
    'bfloat16'
    if device_type == 'cuda' and torch.cuda.is_bf16_supported()
    else 'float16'
    if device_type == 'cuda'
    else 'float32'
)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
from contextlib import nullcontext
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

os.makedirs(args.out_dir, exist_ok=True)

# ── Load meta (vocab size etc.) ───────────────────────────────────────────
meta_path = os.path.join(args.data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    print(f"ERROR: {meta_path} not found. Run `python data_prepare.py` first.")
    sys.exit(1)

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
print(f"Vocab size: {vocab_size}")

# ── Data loader ───────────────────────────────────────────────────────────
train_data = np.memmap(os.path.join(args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data   = np.memmap(os.path.join(args.data_dir, 'val.bin'),   dtype=np.uint16, mode='r')
print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i    : i + args.block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1: i + args.block_size + 1].astype(np.int64)) for i in ix
    ])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ── Model init ────────────────────────────────────────────────────────────
iter_num      = 0
best_val_loss = float('inf')

model_args = dict(
    block_size=args.block_size,
    vocab_size=vocab_size,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_embd=args.n_embd,
    dropout=args.dropout,
    bias=args.bias,
)

if args.init_from == 'scratch':
    print("Initialising model from scratch...")
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

elif args.init_from == 'resume':
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    print(f"Resuming from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Force architecture to match checkpoint
    for k in ['block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'bias']:
        model_args[k] = checkpoint['model_args'][k]
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)
    state_dict = checkpoint['model']
    # Strip any unwanted prefix (from torch.compile)
    unwanted = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed at iter {iter_num}, best_val_loss={best_val_loss:.4f}")

model.to(device)

# Optional torch.compile (PyTorch 2.0+)
if args.compile:
    print("Compiling model with torch.compile()…")
    model = torch.compile(model)

# ── Optimiser & scaler ────────────────────────────────────────────────────
scaler    = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    betas=(args.beta1, args.beta2),
    device_type=device_type,
)
if args.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])


# ── LR schedule ───────────────────────────────────────────────────────────
def get_lr(it):
    # Linear warmup
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    # After decay: hold at min_lr
    if it > args.max_iters:
        return args.min_lr
    # Cosine decay
    ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


# ── Evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(iter_num, best_val_loss, val_loss, filename='ckpt.pt'):
    raw_model = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'model':       raw_model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'model_args':  model_args,
        'iter_num':    iter_num,
        'best_val_loss': best_val_loss,
        'val_loss':    val_loss,
        'config':      gpt_config,
    }
    path = os.path.join(args.out_dir, filename)
    torch.save(checkpoint, path)
    return path


# ── Training loop ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Training NanoGPT-Arabic")
print(f"  Layers: {args.n_layer} | Heads: {args.n_head} | Embd: {args.n_embd}")
print(f"  Block: {args.block_size} | Batch: {args.batch_size}")
print(f"  Max iters: {args.max_iters} | LR: {args.learning_rate}")
print(f"  Params: {model.get_num_params()/1e6:.2f}M")
print(f"{'='*60}\n")

X, Y = get_batch('train')
t0 = time.time()
running_mfu = -1.0

while iter_num <= args.max_iters:

    # ── LR update ────────────────────────────────────────────────────────
    lr = get_lr(iter_num) if args.decay_lr else args.learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # ── Eval & checkpoint ─────────────────────────────────────────────────
    if iter_num % args.eval_interval == 0:
        losses = estimate_loss()
        tl, vl = losses['train'], losses['val']
        train_ppl = math.exp(tl)
        val_ppl   = math.exp(vl)
        print(f"\n[Eval] iter {iter_num:5d} | "
              f"train loss {tl:.4f} (ppl {train_ppl:.1f}) | "
              f"val loss {vl:.4f} (ppl {val_ppl:.1f})")

        if vl < best_val_loss or args.always_save_ckpt:
            if vl < best_val_loss:
                best_val_loss = vl
            if iter_num > 0:
                path = save_checkpoint(iter_num, best_val_loss, vl)
                # Also save a separate best_gpt_model.pt for the app
                best_path = save_checkpoint(iter_num, best_val_loss, vl,
                                            filename='best_gpt_model.pt')
                print(f"         Checkpoint saved → {path}")

    if iter_num == 0:
        # Log model size once before first step
        iter_num += 1
        continue

    # ── Forward / backward ────────────────────────────────────────────────
    with ctx:
        _, loss = model(X, Y)

    X, Y = get_batch('train')   # prefetch next batch

    scaler.scale(loss).backward()

    if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ── Logging ───────────────────────────────────────────────────────────
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % args.log_interval == 0:
        lossf = loss.item()
        print(f"iter {iter_num:5d} | loss {lossf:.4f} | "
              f"lr {lr:.2e} | {dt*1000:.0f}ms/iter")

    iter_num += 1

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} "
      f"(PPL: {math.exp(best_val_loss):.1f})")
print(f"Checkpoint saved to: {args.out_dir}/")
