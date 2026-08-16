# -*- coding: utf-8 -*-
"""
FILE: Switch_GShard_MOE_SAFE_FINAL.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import time
import csv
import json
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DistilBertModel
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
MAX_LEN = 128

NUM_EXPERTS = 4
DENSE_MULTIPLIER = 8
MOE_EXPERT_MULTIPLIER = 8

USE_SMALL_DEBUG_SUBSET = False
USE_BALANCED_SUBSET = True

EARLY_STOPPING_PATIENCE = 2
GRAD_CLIP = 1.0
AUX_LOSS_WEIGHT = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

# SAFER SETTINGS FOR REMOTE SERVER / MobaXterm
NUM_WORKERS_TRAIN = 0
NUM_WORKERS_VAL = 0
PIN_MEMORY = torch.cuda.is_available()

# ============================================================
# ONLY GSHARD MEMORY FIX CONFIG
# ============================================================

GSHARD_BATCH_SIZE = 4
GSHARD_GRAD_ACCUM_STEPS = 4

RUN_TAG = time.strftime("%Y%m%d_%H%M%S")

ROOT_DIR = f"SG_outputs_safe_{RUN_TAG}"
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "SG_checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "SG_plots")
LOGS_DIR = os.path.join(ROOT_DIR, "SG_logs")
CSV_DIR = os.path.join(ROOT_DIR, "SG_csv")
SUMMARY_DIR = os.path.join(ROOT_DIR, "SG_summary")

for d in [ROOT_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR, CSV_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# LOGGER
# ============================================================

LOG_FILE = os.path.join(LOGS_DIR, f"SG_train_log_safe_{RUN_TAG}.txt")

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def isatty(self):
        return False

log_f = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

print("=" * 90)
print("SAFE HETEROGENEOUS NLI: DENSE vs SWITCH vs GSHARD")
print("=" * 90)
print(f"Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"GShard Batch Size: {GSHARD_BATCH_SIZE}")
print(f"GShard Grad Accum Steps: {GSHARD_GRAD_ACCUM_STEPS}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LR}")
print(f"Max Length: {MAX_LEN}")
print(f"Experts: {NUM_EXPERTS}")
print(f"Run Tag: {RUN_TAG}")
print(f"Root Dir: {ROOT_DIR}")
print(f"AMP Enabled: {USE_AMP}")
print("=" * 90)

# ============================================================
# AMP HELPERS
# ============================================================

def autocast_context():
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", enabled=USE_AMP)
    else:
        return torch.amp.autocast(device_type="cpu", enabled=False)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP) if torch.cuda.is_available() else None

# ============================================================
# TOKENIZER
# ============================================================

print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(
        example["premise"],
        example["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

# ============================================================
# DATASET PREP
# ============================================================

def filter_valid(example):
    return example["label"] != -1

def add_source(example, source_id):
    example["source_id"] = source_id
    return example

def balanced_select(dataset, n):
    return dataset.shuffle(seed=SEED).select(range(min(n, len(dataset))))

def load_and_prepare_datasets():
    print("\n[INFO] Loading SNLI...")
    snli = load_dataset("snli")

    print("[INFO] Loading MultiNLI...")
    mnli = load_dataset("multi_nli")

    print("[INFO] Loading ANLI...")
    anli = load_dataset("anli")

    snli_train = snli["train"].filter(filter_valid)
    snli_val = snli["validation"].filter(filter_valid)

    mnli_train = mnli["train"].filter(filter_valid)
    mnli_val_matched = mnli["validation_matched"].filter(filter_valid)
    mnli_val_mismatched = mnli["validation_mismatched"].filter(filter_valid)

    anli_train = concatenate_datasets([
        anli["train_r1"].filter(filter_valid),
        anli["train_r2"].filter(filter_valid),
        anli["train_r3"].filter(filter_valid)
    ])

    anli_val = concatenate_datasets([
        anli["dev_r1"].filter(filter_valid),
        anli["dev_r2"].filter(filter_valid),
        anli["dev_r3"].filter(filter_valid)
    ])

    snli_train = snli_train.map(lambda x: add_source(x, 0))
    snli_val = snli_val.map(lambda x: add_source(x, 0))

    mnli_train = mnli_train.map(lambda x: add_source(x, 1))
    mnli_val_matched = mnli_val_matched.map(lambda x: add_source(x, 1))
    mnli_val_mismatched = mnli_val_mismatched.map(lambda x: add_source(x, 1))

    anli_train = anli_train.map(lambda x: add_source(x, 2))
    anli_val = anli_val.map(lambda x: add_source(x, 2))

    if USE_BALANCED_SUBSET:
        print("[INFO] Using BALANCED training subset for fair MoE comparison")
        subset_size = 100000 if not USE_SMALL_DEBUG_SUBSET else 20000
        snli_train = balanced_select(snli_train, subset_size)
        mnli_train = balanced_select(mnli_train, subset_size)
        anli_train = balanced_select(anli_train, subset_size)

    if USE_SMALL_DEBUG_SUBSET:
        print("[INFO] Using SMALL DEBUG SUBSET")
        snli_val = balanced_select(snli_val, 5000)
        mnli_val_matched = balanced_select(mnli_val_matched, 5000)
        mnli_val_mismatched = balanced_select(mnli_val_mismatched, 5000)
        anli_val = balanced_select(anli_val, 5000)

    train_combined = concatenate_datasets([snli_train, mnli_train, anli_train]).shuffle(seed=SEED)

    print("[INFO] Tokenizing datasets...")
    train_combined = train_combined.map(preprocess, batched=True)
    snli_val = snli_val.map(preprocess, batched=True)
    mnli_val_matched = mnli_val_matched.map(preprocess, batched=True)
    mnli_val_mismatched = mnli_val_mismatched.map(preprocess, batched=True)
    anli_val = anli_val.map(preprocess, batched=True)

    cols = ["input_ids", "attention_mask", "label", "source_id"]

    train_combined.set_format(type="torch", columns=cols)
    snli_val.set_format(type="torch", columns=cols)
    mnli_val_matched.set_format(type="torch", columns=cols)
    mnli_val_mismatched.set_format(type="torch", columns=cols)
    anli_val.set_format(type="torch", columns=cols)

    return train_combined, snli_val, mnli_val_matched, mnli_val_mismatched, anli_val

train_ds, snli_val_ds, mnli_matched_ds, mnli_mismatched_ds, anli_val_ds = load_and_prepare_datasets()

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS_TRAIN,
    pin_memory=PIN_MEMORY
)

gshard_train_loader = DataLoader(
    train_ds,
    batch_size=GSHARD_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS_TRAIN,
    pin_memory=PIN_MEMORY
)

snli_val_loader = DataLoader(snli_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY)
mnli_matched_loader = DataLoader(mnli_matched_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY)
mnli_mismatched_loader = DataLoader(mnli_mismatched_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY)
anli_val_loader = DataLoader(anli_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY)

print("\n[INFO] Dataset Ready")
print(f"Train batches: {len(train_loader)}")
print(f"GShard train batches: {len(gshard_train_loader)}")

# ============================================================
# CSV INIT
# ============================================================

BASELINE_CSV = os.path.join(CSV_DIR, f"baseline_metrics_{RUN_TAG}.csv")
SWITCH_CSV = os.path.join(CSV_DIR, f"switch_metrics_{RUN_TAG}.csv")
GSHARD_CSV = os.path.join(CSV_DIR, f"gshard_metrics_{RUN_TAG}.csv")

FINAL_TXT = os.path.join(SUMMARY_DIR, f"final_summary_{RUN_TAG}.txt")
FINAL_JSON = os.path.join(SUMMARY_DIR, f"final_summary_{RUN_TAG}.json")

def init_csv(path, is_moe=False):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "epoch", "train_loss", "train_accuracy",
                "snli_val_acc", "mnli_matched_val_acc", "mnli_mismatched_val_acc", "anli_val_acc",
                "epoch_time_sec"
            ]
            if is_moe:
                header += [f"expert_{i}_usage" for i in range(NUM_EXPERTS)]
            writer.writerow(header)

init_csv(BASELINE_CSV, is_moe=False)
init_csv(SWITCH_CSV, is_moe=True)
init_csv(GSHARD_CSV, is_moe=True)

# ============================================================
# UTILS
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, filename)
    print(f"[CHECKPOINT] Saved: {filename}")

def benchmark_inference(model, dataloader, is_moe=False, warmup_steps=10):
    model.eval()

    if hasattr(model, "bert") and hasattr(model.bert, "gradient_checkpointing_disable"):
        model.bert.gradient_checkpointing_disable()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup_steps:
                break
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with autocast_context():
                _ = model(input_ids, attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with autocast_context():
                _ = model(input_ids, attention_mask)

            total_samples += input_ids.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.time()
    total_time = end - start

    latency_per_batch = total_time / len(dataloader)
    latency_per_sample = total_time / total_samples
    throughput = total_samples / total_time

    return latency_per_batch, latency_per_sample, throughput

# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.best = -1
        self.counter = 0

    def step(self, metric):
        if metric > self.best:
            self.best = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self):
        return self.counter >= self.patience

# ============================================================
# MODEL BUILDING BLOCKS
# ============================================================

class DenseFFN(nn.Module):
    def __init__(self, hidden_dim, multiplier=8, dropout=0.2):
        super().__init__()
        ffn_dim = hidden_dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SwitchMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, multiplier=8, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        ffn_dim = hidden_dim * multiplier

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.contiguous().view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top1 = torch.argmax(router_probs, dim=-1)

        expert_counts = torch.bincount(top1, minlength=self.num_experts).float()
        density = expert_counts / (expert_counts.sum() + 1e-9)

        router_mean = router_probs.mean(dim=0).float()
        aux_loss = (router_mean * density.to(router_mean.dtype)).sum() * self.num_experts

        out = torch.zeros(x_flat.shape, device=x_flat.device, dtype=x_flat.dtype)

        for i in range(self.num_experts):
            mask = (top1 == i)
            if mask.sum().item() == 0:
                continue
            expert_out = self.experts[i](x_flat[mask])
            if expert_out.dtype != out.dtype:
                expert_out = expert_out.to(out.dtype)
            out[mask] = expert_out

        return out.view(batch_size, seq_len, hidden_dim), aux_loss, density.detach().cpu()

class GShardMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts=4, multiplier=8, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts)
        ffn_dim = hidden_dim * multiplier

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.contiguous().view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top2_vals, top2_idx = torch.topk(router_probs, k=2, dim=-1)
        top2_vals = top2_vals / (top2_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros(x_flat.shape, device=x_flat.device, dtype=x_flat.dtype)
        usage_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)

        for rank in range(2):
            expert_ids = top2_idx[:, rank]
            weights = top2_vals[:, rank].unsqueeze(-1).to(x_flat.dtype)

            for i in range(self.num_experts):
                mask = (expert_ids == i)
                if mask.sum().item() == 0:
                    continue

                expert_out = self.experts[i](x_flat[mask])

                if expert_out.dtype != out.dtype:
                    expert_out = expert_out.to(out.dtype)

                out[mask] += weights[mask] * expert_out
                usage_counts[i] += mask.sum().float()

        density = usage_counts / (usage_counts.sum() + 1e-9)
        router_mean = router_probs.mean(dim=0).float()
        aux_loss = (router_mean * density.to(router_mean.dtype)).sum() * self.num_experts

        return out.view(batch_size, seq_len, hidden_dim), aux_loss, density.detach().cpu()

# ============================================================
# MODELS
# ============================================================

class DenseBaselineModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_dim = self.bert.config.hidden_size
        self.ffn = DenseFFN(hidden_dim, multiplier=DENSE_MULTIPLIER)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = self.ffn(x)
        cls = x[:, 0, :]
        return self.classifier(cls)

class SwitchModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_dim = self.bert.config.hidden_size
        self.moe = SwitchMoE(hidden_dim, num_experts=NUM_EXPERTS, multiplier=MOE_EXPERT_MULTIPLIER)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x, aux_loss, density = self.moe(x)
        cls = x[:, 0, :]
        logits = self.classifier(cls)
        return logits, aux_loss, density

class GShardModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert.gradient_checkpointing_enable()
        hidden_dim = self.bert.config.hidden_size
        self.moe = GShardMoE(hidden_dim, num_experts=NUM_EXPERTS, multiplier=MOE_EXPERT_MULTIPLIER)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x, aux_loss, density = self.moe(x)
        cls = x[:, 0, :]
        logits = self.classifier(cls)
        return logits, aux_loss, density

# ============================================================
# ACTIVE PARAMS
# ============================================================

def count_active_params_switch(model):
    bert_params = count_parameters(model.bert)
    one_expert = count_parameters(model.moe.experts[0])
    router = count_parameters(model.moe.router)
    classifier = count_parameters(model.classifier)
    return bert_params + one_expert + router + classifier

def count_active_params_gshard(model):
    bert_params = count_parameters(model.bert)
    two_experts = count_parameters(model.moe.experts[0]) * 2
    router = count_parameters(model.moe.router)
    classifier = count_parameters(model.classifier)
    return bert_params + two_experts + router + classifier

# ============================================================
# EVALUATION
# ============================================================

def evaluate_dense(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with autocast_context():
                logits = model(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

def evaluate_moe(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with autocast_context():
                logits, _, _ = model(input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

# ============================================================
# TRAINING
# ============================================================

def train_dense(model, dataloader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 200 == 0:
            print(f"[DENSE][Step {step}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / len(dataloader), correct / total

def train_moe(model, dataloader, optimizer, model_name="SWITCH"):
    model.train()
    total_loss, correct, total = 0, 0, 0
    expert_usage_accum = None

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            logits, aux_loss, density = model(input_ids, attention_mask)
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss + AUX_LOSS_WEIGHT * aux_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        density = density.detach().cpu() if isinstance(density, torch.Tensor) else density
        if expert_usage_accum is None:
            expert_usage_accum = density.clone()
        else:
            expert_usage_accum += density

        if step % 200 == 0:
            print(f"[{model_name}][Step {step}/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_usage = (expert_usage_accum / len(dataloader)).numpy()
    return total_loss / len(dataloader), correct / total, avg_usage

def train_gshard_memory_safe(model, dataloader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    expert_usage_accum = None

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast_context():
            logits, aux_loss, density = model(input_ids, attention_mask)
            ce_loss = F.cross_entropy(logits, labels)
            full_loss = ce_loss + AUX_LOSS_WEIGHT * aux_loss
            loss = full_loss / GSHARD_GRAD_ACCUM_STEPS

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % GSHARD_GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += full_loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        density_cpu = density.detach().cpu() if isinstance(density, torch.Tensor) else density
        if expert_usage_accum is None:
            expert_usage_accum = density_cpu.clone()
        else:
            expert_usage_accum += density_cpu

        if step % 200 == 0:
            print(f"[GSHARD][Step {step}/{len(dataloader)}] Loss: {full_loss.item():.4f}")

        del input_ids, attention_mask, labels
        del logits, aux_loss, density, density_cpu
        del ce_loss, full_loss, loss, preds

        if torch.cuda.is_available() and step % 50 == 0:
            torch.cuda.empty_cache()

    avg_usage = (expert_usage_accum / len(dataloader)).numpy()
    return total_loss / len(dataloader), correct / total, avg_usage

# ============================================================
# SAVE METRICS
# ============================================================

def save_epoch_csv(path, epoch, train_loss, train_acc, snli_acc, mnli_m_acc, mnli_mm_acc, anli_acc, epoch_time, expert_usage=None):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [epoch, train_loss, train_acc, snli_acc, mnli_m_acc, mnli_mm_acc, anli_acc, epoch_time]
        if expert_usage is not None:
            row += [float(x) for x in expert_usage]
        writer.writerow(row)

# ============================================================
# PLOTS
# ============================================================

def plot_metric(histories, metric_key, title, filename, ylabel):
    plt.figure()
    for name, hist in histories.items():
        if len(hist) == 0:
            continue
        epochs = [x["epoch"] for x in hist]
        vals = [x[metric_key] for x in hist]
        plt.plot(epochs, vals, marker='o', label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def plot_expert_usage(history, model_name):
    if len(history) == 0 or "expert_usage" not in history[0]:
        return

    epochs = [x["epoch"] for x in history]
    plt.figure()

    for i in range(NUM_EXPERTS):
        vals = [x["expert_usage"][i] for x in history]
        plt.plot(epochs, vals, marker='o', label=f"Expert {i}")

    plt.xlabel("Epoch")
    plt.ylabel("Average Usage")
    plt.title(f"{model_name} Expert Usage Across Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name.lower()}_expert_usage_{RUN_TAG}.png"))
    plt.close()

def plot_final_bar(labels, values, title, ylabel, filename):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

# ============================================================
# MAIN TRAINING
# ============================================================

def run_dense():
    print("\n" + "="*90)
    print("STEP 1: TRAINING DENSE BASELINE")
    print("="*90)

    model = DenseBaselineModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_path = os.path.join(CHECKPOINT_DIR, f"best_dense_{RUN_TAG}.pt")

    for epoch in range(EPOCHS):
        print("\n" + "-"*90)
        print(f"[DENSE] Epoch {epoch+1}/{EPOCHS}")
        print("-"*90)

        start = time.time()

        train_loss, train_acc = train_dense(model, train_loader, optimizer)

        snli_acc = evaluate_dense(model, snli_val_loader)
        mnli_m_acc = evaluate_dense(model, mnli_matched_loader)
        mnli_mm_acc = evaluate_dense(model, mnli_mismatched_loader)
        anli_acc = evaluate_dense(model, anli_val_loader)

        epoch_time = time.time() - start
        monitor_metric = (snli_acc + mnli_m_acc + mnli_mm_acc + anli_acc) / 4

        print(f"[DENSE] Train Loss: {train_loss:.4f}")
        print(f"[DENSE] Train Accuracy: {train_acc:.4f}")
        print(f"[DENSE] SNLI Val: {snli_acc:.4f}")
        print(f"[DENSE] MNLI Matched: {mnli_m_acc:.4f}")
        print(f"[DENSE] MNLI Mismatched: {mnli_mm_acc:.4f}")
        print(f"[DENSE] ANLI Val: {anli_acc:.4f}")
        print(f"[DENSE] Epoch Time: {epoch_time:.2f} sec")

        save_epoch_csv(BASELINE_CSV, epoch+1, train_loss, train_acc, snli_acc, mnli_m_acc, mnli_mm_acc, anli_acc, epoch_time)

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "snli_acc": snli_acc,
            "mnli_matched_acc": mnli_m_acc,
            "mnli_mismatched_acc": mnli_mm_acc,
            "anli_acc": anli_acc,
            "epoch_time": epoch_time
        })

        save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, f"dense_epoch_{epoch+1}_{RUN_TAG}.pt"))

        if early_stopper.step(monitor_metric):
            torch.save(model.state_dict(), best_path)
            print(f"[DENSE] Best model updated -> {best_path}")

        if early_stopper.should_stop():
            print("[DENSE] Early stopping triggered")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history

def run_switch():
    print("\n" + "="*90)
    print("STEP 2: TRAINING SWITCH MoE")
    print("="*90)

    model = SwitchModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_path = os.path.join(CHECKPOINT_DIR, f"best_switch_{RUN_TAG}.pt")

    for epoch in range(EPOCHS):
        print("\n" + "-"*90)
        print(f"[SWITCH] Epoch {epoch+1}/{EPOCHS}")
        print("-"*90)

        start = time.time()

        train_loss, train_acc, expert_usage = train_moe(model, train_loader, optimizer, "SWITCH")

        snli_acc = evaluate_moe(model, snli_val_loader)
        mnli_m_acc = evaluate_moe(model, mnli_matched_loader)
        mnli_mm_acc = evaluate_moe(model, mnli_mismatched_loader)
        anli_acc = evaluate_moe(model, anli_val_loader)

        epoch_time = time.time() - start
        monitor_metric = (snli_acc + mnli_m_acc + mnli_mm_acc + anli_acc) / 4

        print(f"[SWITCH] Train Loss: {train_loss:.4f}")
        print(f"[SWITCH] Train Accuracy: {train_acc:.4f}")
        print(f"[SWITCH] SNLI Val: {snli_acc:.4f}")
        print(f"[SWITCH] MNLI Matched: {mnli_m_acc:.4f}")
        print(f"[SWITCH] MNLI Mismatched: {mnli_mm_acc:.4f}")
        print(f"[SWITCH] ANLI Val: {anli_acc:.4f}")
        print(f"[SWITCH] Expert Usage: {expert_usage}")
        print(f"[SWITCH] Epoch Time: {epoch_time:.2f} sec")

        save_epoch_csv(SWITCH_CSV, epoch+1, train_loss, train_acc, snli_acc, mnli_m_acc, mnli_mm_acc, anli_acc, epoch_time, expert_usage)

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "snli_acc": snli_acc,
            "mnli_matched_acc": mnli_m_acc,
            "mnli_mismatched_acc": mnli_mm_acc,
            "anli_acc": anli_acc,
            "expert_usage": expert_usage.tolist(),
            "epoch_time": epoch_time
        })

        save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, f"switch_epoch_{epoch+1}_{RUN_TAG}.pt"))

        if early_stopper.step(monitor_metric):
            torch.save(model.state_dict(), best_path)
            print(f"[SWITCH] Best model updated -> {best_path}")

        if early_stopper.should_stop():
            print("[SWITCH] Early stopping triggered")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history

def run_gshard():
    print("\n" + "="*90)
    print("STEP 3: TRAINING GSHARD MoE")
    print("="*90)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = GShardModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"[GSHARD] Using smaller batch size = {GSHARD_BATCH_SIZE}")
    print(f"[GSHARD] Gradient accumulation steps = {GSHARD_GRAD_ACCUM_STEPS}")

    history = []
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    best_path = os.path.join(CHECKPOINT_DIR, f"best_gshard_{RUN_TAG}.pt")

    for epoch in range(EPOCHS):
        print("\n" + "-"*90)
        print(f"[GSHARD] Epoch {epoch+1}/{EPOCHS}")
        print("-"*90)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = time.time()

        train_loss, train_acc, expert_usage = train_gshard_memory_safe(
            model,
            gshard_train_loader,
            optimizer
        )

        snli_acc = evaluate_moe(model, snli_val_loader)
        mnli_m_acc = evaluate_moe(model, mnli_matched_loader)
        mnli_mm_acc = evaluate_moe(model, mnli_mismatched_loader)
        anli_acc = evaluate_moe(model, anli_val_loader)

        epoch_time = time.time() - start
        monitor_metric = (snli_acc + mnli_m_acc + mnli_mm_acc + anli_acc) / 4

        print(f"[GSHARD] Train Loss: {train_loss:.4f}")
        print(f"[GSHARD] Train Accuracy: {train_acc:.4f}")
        print(f"[GSHARD] SNLI Val: {snli_acc:.4f}")
        print(f"[GSHARD] MNLI Matched: {mnli_m_acc:.4f}")
        print(f"[GSHARD] MNLI Mismatched: {mnli_mm_acc:.4f}")
        print(f"[GSHARD] ANLI Val: {anli_acc:.4f}")
        print(f"[GSHARD] Expert Usage: {expert_usage}")
        print(f"[GSHARD] Epoch Time: {epoch_time:.2f} sec")

        save_epoch_csv(GSHARD_CSV, epoch+1, train_loss, train_acc, snli_acc, mnli_m_acc, mnli_mm_acc, anli_acc, epoch_time, expert_usage)

        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "snli_acc": snli_acc,
            "mnli_matched_acc": mnli_m_acc,
            "mnli_mismatched_acc": mnli_mm_acc,
            "anli_acc": anli_acc,
            "expert_usage": expert_usage.tolist(),
            "epoch_time": epoch_time
        })

        save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, f"gshard_epoch_{epoch+1}_{RUN_TAG}.pt"))

        if early_stopper.step(monitor_metric):
            torch.save(model.state_dict(), best_path)
            print(f"[GSHARD] Best model updated -> {best_path}")

        if early_stopper.should_stop():
            print("[GSHARD] Early stopping triggered")
            break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history

# ============================================================
# MAIN
# ============================================================

def main():
    dense_model, dense_hist = run_dense()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    switch_model, switch_hist = run_switch()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gshard_model, gshard_hist = run_gshard()

    print("\n" + "="*90)
    print("STEP 4: FINAL COMPARISON")
    print("="*90)

    dense_total = count_parameters(dense_model)
    switch_total = count_parameters(switch_model)
    gshard_total = count_parameters(gshard_model)

    dense_active = dense_total
    switch_active = count_active_params_switch(switch_model)
    gshard_active = count_active_params_gshard(gshard_model)

    print(f"DENSE Total Params:   {dense_total}")
    print(f"DENSE Active Params:  {dense_active}")
    print(f"SWITCH Total Params:  {switch_total}")
    print(f"SWITCH Active Params: {switch_active}")
    print(f"GSHARD Total Params:  {gshard_total}")
    print(f"GSHARD Active Params: {gshard_active}")

    print("\n[INFO] Benchmarking inference fairly (same eval batch size for all models)...")

    dense_lat_batch, dense_lat_sample, dense_tp = benchmark_inference(dense_model, mnli_matched_loader, is_moe=False)
    switch_lat_batch, switch_lat_sample, switch_tp = benchmark_inference(switch_model, mnli_matched_loader, is_moe=True)
    gshard_lat_batch, gshard_lat_sample, gshard_tp = benchmark_inference(gshard_model, mnli_matched_loader, is_moe=True)

    print(f"\nDENSE Latency / Batch:  {dense_lat_batch:.6f} sec")
    print(f"DENSE Latency / Sample: {dense_lat_sample:.6f} sec")
    print(f"DENSE Throughput:       {dense_tp:.2f} samples/sec")

    print(f"\nSWITCH Latency / Batch:  {switch_lat_batch:.6f} sec")
    print(f"SWITCH Latency / Sample: {switch_lat_sample:.6f} sec")
    print(f"SWITCH Throughput:       {switch_tp:.2f} samples/sec")

    print(f"\nGSHARD Latency / Batch:  {gshard_lat_batch:.6f} sec")
    print(f"GSHARD Latency / Sample: {gshard_lat_sample:.6f} sec")
    print(f"GSHARD Throughput:       {gshard_tp:.2f} samples/sec")

    final_results = {
        "dense_final": dense_hist[-1] if len(dense_hist) > 0 else {},
        "switch_final": switch_hist[-1] if len(switch_hist) > 0 else {},
        "gshard_final": gshard_hist[-1] if len(gshard_hist) > 0 else {},

        "dense_total_params": dense_total,
        "dense_active_params": dense_active,

        "switch_total_params": switch_total,
        "switch_active_params": switch_active,

        "gshard_total_params": gshard_total,
        "gshard_active_params": gshard_active,

        "dense_latency_per_batch": dense_lat_batch,
        "dense_latency_per_sample": dense_lat_sample,
        "dense_throughput": dense_tp,

        "switch_latency_per_batch": switch_lat_batch,
        "switch_latency_per_sample": switch_lat_sample,
        "switch_throughput": switch_tp,

        "gshard_latency_per_batch": gshard_lat_batch,
        "gshard_latency_per_sample": gshard_lat_sample,
        "gshard_throughput": gshard_tp,

        "note_training_fairness": (
            "Dense and Switch trained with batch size 16. "
            "GShard trained with micro-batch 4 + grad accumulation 4 for memory safety. "
            "Hence training time is not strictly apples-to-apples, but inference benchmarking is fair."
        )
    }

    with open(FINAL_TXT, "w", encoding="utf-8") as f:
        f.write("FINAL COMPARISON SUMMARY\n")
        f.write("="*60 + "\n")
        for k, v in final_results.items():
            f.write(f"{k}: {v}\n")

    with open(FINAL_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n[INFO] Final summary saved to:")
    print(FINAL_TXT)
    print(FINAL_JSON)

    histories = {
        "Dense": dense_hist,
        "Switch": switch_hist,
        "GShard": gshard_hist
    }

    print("\n[INFO] Saving plots...")

    plot_metric(histories, "train_acc", "Train Accuracy Comparison", f"train_acc_{RUN_TAG}.png", "Train Accuracy")
    plot_metric(histories, "train_loss", "Train Loss Comparison", f"train_loss_{RUN_TAG}.png", "Train Loss")

    plot_metric(histories, "snli_acc", "SNLI Validation Accuracy", f"snli_val_acc_{RUN_TAG}.png", "Accuracy")
    plot_metric(histories, "mnli_matched_acc", "MNLI Matched Validation Accuracy", f"mnli_matched_acc_{RUN_TAG}.png", "Accuracy")
    plot_metric(histories, "mnli_mismatched_acc", "MNLI Mismatched Validation Accuracy", f"mnli_mismatched_acc_{RUN_TAG}.png", "Accuracy")
    plot_metric(histories, "anli_acc", "ANLI Validation Accuracy", f"anli_acc_{RUN_TAG}.png", "Accuracy")

    plot_expert_usage(switch_hist, "Switch")
    plot_expert_usage(gshard_hist, "GShard")

    plot_final_bar(
        ["Dense Total", "Dense Active", "Switch Total", "Switch Active", "GShard Total", "GShard Active"],
        [dense_total, dense_active, switch_total, switch_active, gshard_total, gshard_active],
        "Total vs Active Parameter Comparison",
        "Parameters",
        f"param_comparison_{RUN_TAG}.png"
    )

    plot_final_bar(
        ["Dense", "Switch", "GShard"],
        [dense_lat_sample, switch_lat_sample, gshard_lat_sample],
        "Latency Comparison (Per Sample)",
        "Seconds / Sample",
        f"latency_per_sample_comparison_{RUN_TAG}.png"
    )

    plot_final_bar(
        ["Dense", "Switch", "GShard"],
        [dense_tp, switch_tp, gshard_tp],
        "Throughput Comparison",
        "Samples / Second",
        f"throughput_comparison_{RUN_TAG}.png"
    )

    plot_final_bar(
        ["Dense", "Switch", "GShard"],
        [
            dense_hist[-1]["epoch_time"] if len(dense_hist) > 0 else 0,
            switch_hist[-1]["epoch_time"] if len(switch_hist) > 0 else 0,
            gshard_hist[-1]["epoch_time"] if len(gshard_hist) > 0 else 0
        ],
        "Training Epoch Time Comparison (Interpret Carefully)",
        "Seconds",
        f"epoch_time_comparison_{RUN_TAG}.png"
    )

    print(f"[INFO] All plots saved in: {PLOTS_DIR}")

    print("\n" + "="*90)
    print("TRAINING + SAFE FAIR COMPARISON COMPLETED SUCCESSFULLY")
    print("="*90)

if __name__ == "__main__":
    main()