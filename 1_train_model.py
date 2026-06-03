
import os
import math
import copy
import random
import warnings
from dataclasses import dataclass
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
@dataclass
class CFG:
    seed: int = 42
    data_dir: str = "dataset_final_split"

    x_train_file: str = "X_train.npy"
    y_train_file: str = "y_train.npy"
    x_val_file: str = "X_val.npy"
    y_val_file: str = "y_val.npy"

    x_tl_file: str = "X_tl_train.npy"
    y_tl_file: str = "y_tl_train.npy"

    x_test_file: str = "X_test.npy"
    y_test_file: str = "y_test.npy"

    base_ckpt: str = "cnn_v13_base_best.pth"
    tl_head_ckpt: str = "cnn_v13_tl_head_best.pth"
    tl_partial_ckpt: str = "cnn_v13_tl_partial_best.pth"
    tl_full_ckpt: str = "cnn_v13_tl_full_best.pth"
    final_tl_ckpt: str = "cnn_v13_tl_partial_best.pth"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    base_epochs: int = 120
    base_batch_size: int = 64
    base_lr: float = 8e-4
    base_weight_decay: float = 1e-4
    base_patience: int = 18

    tl_val_ratio: float = 0.30
    shuffle_tl_split: bool = True

    tl_head_epochs: int = 50
    tl_head_batch_size: int = 32
    tl_head_lr: float = 6e-4
    tl_head_weight_decay: float = 1e-4
    tl_head_patience: int = 14

    tl_partial_epochs: int = 60
    tl_partial_batch_size: int = 32
    tl_partial_lr: float = 1.2e-4
    tl_partial_weight_decay: float = 5e-5
    tl_partial_patience: int = 16

    tl_full_epochs: int = 90
    tl_full_batch_size: int = 48
    tl_full_lr: float = 5e-5
    tl_full_weight_decay: float = 3e-5
    tl_full_patience: int = 20

    grad_clip: float = 1.5

    test_batch_size: int = 32

    use_input_norm: bool = True
    use_target_norm: bool = True
    eps: float = 1e-6

    use_tl_augmentation: bool = False
    aug_noise_std: float = 0.008
    aug_scale_min: float = 0.985
    aug_scale_max: float = 1.015
    aug_channel_dropout_prob: float = 0.04
    aug_time_mask_prob: float = 0.06
    aug_time_mask_ratio: float = 0.04
    aug_random_shift_prob: float = 0.15
    aug_random_shift_max: int = 4

    use_tta: bool = False
    tta_passes: int = 5
    tta_noise_std: float = 0.004
    tta_scale_min: float = 0.992
    tta_scale_max: float = 1.008

    use_amp: bool = torch.cuda.is_available()
    print_every: int = 1

    # Hyperparameters for weighted loss
    use_weighted_loss: bool = True
    loss_weights: tuple = (1.5, 1.0, 1.0)   # prioritize dim0 a bit more
    loss_beta: float = 1.0

    # Control flags for TL stages
    run_tl_partial: bool = True
    run_tl_full: bool = False


cfg = CFG()


class TargetStandardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = y.mean(axis=0)
        self.std_ = y.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.float32)
        return y * self.std_ + self.mean_


def inverse_transform_torch(y, scaler):
    mean = torch.tensor(scaler.mean_, dtype=y.dtype, device=y.device)
    std = torch.tensor(scaler.std_, dtype=y.dtype, device=y.device)
    return y * std + mean


# =========================================================
# SEED
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(cfg.seed)


# =========================================================
# UTILS
# =========================================================
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def summarize_history(history, stage_name="STAGE"):
    if history is None or len(history) == 0:
        print(f"[{stage_name}] No history found.")
        return None, None

    df = pd.DataFrame(history)

    # بهترین epoch بر اساس val_monitor_score
    if "val_monitor_score" in df.columns:
        best_idx = df["val_monitor_score"].astype(float).idxmin()
    elif "val_rmse_3d" in df.columns:
        best_idx = df["val_rmse_3d"].astype(float).idxmin()
    elif "val_rmse" in df.columns:
        best_idx = df["val_rmse"].astype(float).idxmin()
    else:
        best_idx = df.index[-1]

    best_row = df.loc[best_idx]

    print("\n" + "=" * 80)
    print(f"{stage_name} STAGE METRICS SUMMARY")
    print("=" * 80)
    print(
        f"Best epoch            : {int(best_row['epoch']) if 'epoch' in df.columns else best_idx}")

    if "train_rmse" in df.columns:
        print(f"Best train RMSE       : {best_row['train_rmse']:.6f}")
    if "val_rmse" in df.columns:
        print(f"Best val RMSE         : {best_row['val_rmse']:.6f}")
    if "val_rmse_3d" in df.columns:
        print(f"Best val 3D RMSE      : {best_row['val_rmse_3d']:.6f}")
    if "train_mae" in df.columns:
        print(f"Best train MAE        : {best_row['train_mae']:.6f}")
    if "val_mae" in df.columns:
        print(f"Best val MAE          : {best_row['val_mae']:.6f}")
    if "val_euc_mean" in df.columns:
        print(f"Best val mean Euc Err : {best_row['val_euc_mean']:.6f}")
    if "lr" in df.columns:
        print(f"Learning rate         : {best_row['lr']:.6e}")

    return df, best_row


def rmse_3d(y_true, y_pred):
    """
    3D localization RMSE:
    sqrt(mean((dx^2 + dy^2 + dz^2)))
    """
    return float(np.sqrt(np.mean(np.sum((y_true - y_pred) ** 2, axis=1))))


def load_npy_pair(x_path, y_path):
    ensure_exists(x_path)
    ensure_exists(y_path)
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def print_shape(name, arr):
    print(f"{name:>16s}: shape={arr.shape}, dtype={arr.dtype}")


def regression_metrics(y_true, y_pred, prefix=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    out = {f"{prefix}mae": mae, f"{prefix}rmse": rmse}

    if y_true.ndim == 2 and y_true.shape[1] in [2, 3]:
        euc = np.linalg.norm(y_true - y_pred, axis=1)
        out[f"{prefix}euc_mean"] = float(np.mean(euc))
        out[f"{prefix}euc_median"] = float(np.median(euc))
        out[f"{prefix}rmse_3d"] = rmse_3d(y_true, y_pred)

        for d in range(y_true.shape[1]):
            out[f"{prefix}dim{d}_mae"] = float(
                np.mean(np.abs(y_true[:, d] - y_pred[:, d])))
            out[f"{prefix}dim{d}_rmse"] = float(
                np.sqrt(np.mean((y_true[:, d] - y_pred[:, d]) ** 2)))
            out[f"{prefix}dim{d}_bias"] = float(
                np.mean(y_pred[:, d] - y_true[:, d]))

    return out


# =========================================================
# NORMALIZERS
# =========================================================


class ChannelWiseInputNormalizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = np.maximum(
            X.std(axis=(0, 2), keepdims=True).astype(np.float32), self.eps)

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =========================================================
# DATASET
# =========================================================
class SignalDataset(Dataset):
    def __init__(self, X, y=None, augment=False, is_tl=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        if X.shape[1] > X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
        self.X = X
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.augment = augment
        self.is_tl = is_tl

    def __len__(self):
        return len(self.X)

    def _augment_signal(self, x):
        if not (self.augment and self.is_tl and cfg.use_tl_augmentation):
            return x
        # Gaussian noise
        if np.random.rand() < 0.75:
            x = x + np.random.normal(0.0, cfg.aug_noise_std,
                                     x.shape).astype(np.float32)
        # Global scale
        if np.random.rand() < 0.6:
            scale = np.random.uniform(cfg.aug_scale_min, cfg.aug_scale_max)
            x = x * scale
        # Random temporal shift
        if np.random.rand() < cfg.aug_random_shift_prob:
            shift = np.random.randint(-cfg.aug_random_shift_max,
                                      cfg.aug_random_shift_max + 1)
            x = np.roll(x, shift, axis=1)
        # Channel dropout
        if np.random.rand() < cfg.aug_channel_dropout_prob:
            c = np.random.randint(0, x.shape[0])
            x[c] *= 0.5
        # Time masking
        if np.random.rand() < cfg.aug_time_mask_prob:
            L = x.shape[1]
            mask_len = max(1, int(L * cfg.aug_time_mask_ratio))
            start = np.random.randint(0, max(1, L - mask_len + 1))
            x[:, start:start + mask_len] *= 0.5
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        x = self._augment_signal(x)
        x = torch.tensor(x, dtype=torch.float32)

        if self.y is None:
            return x

        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# =========================================================
# MODEL
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, p_drop=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, p_drop=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class CNNV13Improved(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, 7, p_drop=0.04),
            nn.MaxPool1d(2),
            ConvBlock(32, 64, 5, p_drop=0.06),
            ResidualConvBlock(64, 3, p_drop=0.04),
            nn.MaxPool1d(2),
            ConvBlock(64, 128, 5, p_drop=0.08),
            ResidualConvBlock(128, 3, p_drop=0.05),
            nn.MaxPool1d(2),
            ConvBlock(128, 256, 3, p_drop=0.10),
            ResidualConvBlock(256, 3, p_drop=0.06),
        )
        self.global_pool_avg = nn.AdaptiveAvgPool1d(1)
        self.global_pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.12),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.features(x)
        avg_feat = self.global_pool_avg(x).squeeze(-1)
        max_feat = self.global_pool_max(x).squeeze(-1)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out = self.head(feat)
        return out


# =========================================================
# FREEZE / UNFREEZE
# =========================================================
def freeze_feature_extractor(model):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True


def unfreeze_last_feature_stage(model):
    """
    Unfreeze only the last 2 modules in model.features.
    In this architecture, features is a Sequential of 8 modules.
    We'll unfreeze modules from index 6 onward for a gradual TL.
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head.parameters():
        p.requires_grad = True

    for idx, module in enumerate(model.features):
        if idx >= 6:
            for p in module.parameters():
                p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# =========================================================
# TRAIN / VALID
# =========================================================
def run_epoch(model, loader, criterion, optimizer=None, scaler=None, train=True, target_scaler=None):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds_all_real = []
    targets_all_real = []

    for batch in loader:
        x, y = batch
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if cfg.use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(x)
                    loss = criterion(preds, y)
            else:
                preds = model(x)
                loss = criterion(preds, y)

            if train:
                optimizer.zero_grad(set_to_none=True)

                if cfg.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip)
                    optimizer.step()

        losses.append(loss.item())

        if target_scaler is not None:
            preds_real = inverse_transform_torch(preds.detach(), target_scaler)
            targets_real = inverse_transform_torch(y.detach(), target_scaler)
        else:
            preds_real = preds.detach()
            targets_real = y.detach()

        preds_all_real.append(preds_real.cpu().numpy())
        targets_all_real.append(targets_real.cpu().numpy())

    preds_all_real = np.concatenate(preds_all_real, axis=0)
    targets_all_real = np.concatenate(targets_all_real, axis=0)

    metrics = regression_metrics(targets_all_real, preds_all_real)
    avg_loss = float(np.mean(losses))
    return avg_loss, metrics, preds_all_real, targets_all_real


# =========================================================
# TTA
# =========================================================
def tta_predict(model, loader):
    model.eval()
    all_preds = []

    for _ in range(cfg.tta_passes):
        preds_one_pass = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.numpy().copy()

                if cfg.use_tta:
                    noise = np.random.normal(
                        0.0, cfg.tta_noise_std, size=x.shape).astype(np.float32)
                    scale = np.random.uniform(
                        cfg.tta_scale_min, cfg.tta_scale_max, size=(
                            x.shape[0], 1, 1)
                    ).astype(np.float32)
                    x = (x + noise) * scale

                x = torch.tensor(x, dtype=torch.float32).to(cfg.device)
                pred = model(x)
                preds_one_pass.append(pred.detach().cpu().numpy())

        preds_one_pass = np.concatenate(preds_one_pass, axis=0)
        all_preds.append(preds_one_pass)

    all_preds = np.stack(all_preds, axis=0)  # (T, N, D)
    return np.mean(all_preds, axis=0)


# =========================================================
# WEIGHTED LOSS
# =========================================================
class WeightedHuberLoss(nn.Module):
    def __init__(self, weights, beta=1.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(
            weights, dtype=torch.float32))
        self.beta = beta

    def forward(self, pred, target):
        diff = pred - target
        abs_diff = diff.abs()

        # Huber / SmoothL1-style loss
        loss = torch.where(
            abs_diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            abs_diff - 0.5 * self.beta
        )

        # apply per-dimension weights
        loss = loss * self.weights.view(1, -1)
        return loss.mean()


# =========================================================
# MODEL SELECTION SCORE
# =========================================================

def choose_monitor_score(metrics):
    # prioritize dim0 a bit, while still keeping 3D RMSE as the main criterion
    if "dim0_rmse" in metrics:
        return 0.7 * metrics.get("rmse_3d", float("inf")) + 0.3 * metrics["dim0_rmse"]
    if "rmse_3d" in metrics:
        return metrics["rmse_3d"]
    if "rmse" in metrics:
        return metrics["rmse"]
    return float("inf")


# =========================================================
# TRAINING LOOP
# =========================================================
def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    epochs,
    lr,
    weight_decay,
    patience,
    ckpt_path,
    stage_name,
    target_scaler=None,
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(4, patience // 4),
        min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_score = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            train=True,
            target_scaler=target_scaler,
        )

        val_loss, val_metrics, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            train=False,
            target_scaler=target_scaler,
        )

        monitor_score = choose_monitor_score(val_metrics)
        scheduler.step(monitor_score)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_monitor_score": monitor_score,
            "lr": optimizer.param_groups[0]["lr"]
        }

        if "euc_mean" in train_metrics:
            row["train_euc_mean"] = train_metrics["euc_mean"]
            row["val_euc_mean"] = val_metrics["euc_mean"]

        if "rmse_3d" in train_metrics:
            row["train_rmse_3d"] = train_metrics["rmse_3d"]
            row["val_rmse_3d"] = val_metrics["rmse_3d"]

        for key in train_metrics:
            if key.startswith("dim"):
                row[f"train_{key}"] = train_metrics[key]

        for key in val_metrics:
            if key.startswith("dim"):
                row[f"val_{key}"] = val_metrics[key]

        history.append(row)

        if epoch % cfg.print_every == 0:
            msg = (
                f"[{stage_name}] Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
                f"train_rmse={train_metrics['rmse']:.4f} | val_rmse={val_metrics['rmse']:.4f}"
            )

            if "rmse_3d" in val_metrics:
                msg += f" | val_rmse_3d={val_metrics['rmse_3d']:.4f}"

            if "euc_mean" in val_metrics:
                msg += f" | val_euc={val_metrics['euc_mean']:.4f}"

            msg += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
            print(msg)

        if monitor_score < best_score:
            best_score = monitor_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"[{stage_name}] Early stopping triggered at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[{stage_name}] Best epoch: {best_epoch} | Best monitor score: {best_score:.4f}")
    return model, history


# =========================================================
# MAIN
# =========================================================
def main():
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)

    cfg = CFG()


# -----------------------
# Load numpy arrays
# -----------------------
    X_train = np.load(os.path.join(cfg.data_dir, cfg.x_train_file))
    y_train = np.load(os.path.join(cfg.data_dir, cfg.y_train_file))

    X_val = np.load(os.path.join(cfg.data_dir, cfg.x_val_file))
    y_val = np.load(os.path.join(cfg.data_dir, cfg.y_val_file))

    X_tl = np.load(os.path.join(cfg.data_dir, cfg.x_tl_file))
    y_tl = np.load(os.path.join(cfg.data_dir, cfg.y_tl_file))

    X_test = np.load(os.path.join(cfg.data_dir, cfg.x_test_file))
    y_test = np.load(os.path.join(cfg.data_dir, cfg.y_test_file))

    # -----------------------------------------------------
    # INPUT NORMALIZATION (fit only on Part I train)
    # -----------------------------------------------------
    input_norm = None
    if cfg.use_input_norm:
        input_norm = ChannelWiseInputNormalizer(eps=cfg.eps)
        X_train = input_norm.fit_transform(X_train)
        X_val = input_norm.transform(X_val)
        X_tl = input_norm.transform(X_tl)
        X_test = input_norm.transform(X_test)
        print("\nApplied channel-wise input normalization using X_train stats.")

    # -----------------------------------------------------
    # TARGET NORMALIZATION (fit only on Part I train)
    # -----------------------------------------------------
    target_scaler = None
    if cfg.use_target_norm:
        target_scaler = TargetStandardizer()
        target_scaler.fit(y_train)

        y_train_norm = target_scaler.transform(y_train)
        y_val_norm = target_scaler.transform(y_val)
        y_tl_norm = target_scaler.transform(y_tl)
        y_test_norm = target_scaler.transform(y_test)

        print("Applied per-dimension target normalization using y_train stats.")
    else:
        y_train_norm = y_train.copy()
        y_val_norm = y_val.copy()
        y_tl_norm = y_tl.copy()
        y_test_norm = y_test.copy()

    # -----------------------------------------------------
    # DATA LOADERS FOR BASE
    # -----------------------------------------------------
    ds_train = SignalDataset(X_train, y_train_norm, augment=False, is_tl=False)
    ds_val = SignalDataset(X_val, y_val_norm, augment=False, is_tl=False)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.base_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.base_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    if cfg.use_weighted_loss:
        criterion = WeightedHuberLoss(
            weights=cfg.loss_weights,
            beta=cfg.loss_beta
        )
        print(
            f"Using Weighted Huber Loss with weights={cfg.loss_weights}, beta={cfg.loss_beta}")
    else:
        criterion = nn.SmoothL1Loss(beta=cfg.loss_beta)
        print(f"Using standard SmoothL1Loss with beta={cfg.loss_beta}")
    criterion = criterion.to(cfg.device)

    in_channels = X_train.shape[1] if X_train.ndim == 3 else 1
    out_dim = y_train.shape[1] if y_train.ndim == 2 else 1

    # -----------------------------------------------------
    # PHASE 1: BASE TRAINING
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: BASE TRAINING ON PART I")
    print("=" * 80)

    model = CNNV13Improved(in_channels=in_channels,
                           out_dim=out_dim).to(cfg.device)

    model, base_history = fit_model(
        model=model,
        train_loader=dl_train,
        val_loader=dl_val,
        criterion=criterion,
        epochs=cfg.base_epochs,
        lr=cfg.base_lr,
        weight_decay=cfg.base_weight_decay,
        patience=cfg.base_patience,
        ckpt_path=cfg.base_ckpt,
        stage_name="BASE",
        target_scaler=target_scaler
    )

    base_df, base_best = summarize_history(base_history, stage_name="BASE")

    # -----------------------------------------------------
    # PHASE 2: TRANSFER LEARNING
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: TRANSFER LEARNING ON PART II")
    print("=" * 80)

    X_tl_train, X_tl_val, y_tl_train_norm, y_tl_val_norm = train_test_split(
        X_tl,
        y_tl_norm,
        test_size=cfg.tl_val_ratio,
        random_state=cfg.seed,
        shuffle=cfg.shuffle_tl_split
    )

    print_shape("X_tl_train", X_tl_train)
    print_shape("y_tl_train", y_tl_train_norm)
    print_shape("X_tl_val", X_tl_val)
    print_shape("y_tl_val", y_tl_val_norm)

    # ---- TL datasets
    ds_tl_train = SignalDataset(
        X_tl_train, y_tl_train_norm, augment=True, is_tl=True)
    ds_tl_val = SignalDataset(X_tl_val, y_tl_val_norm,
                              augment=False, is_tl=True)

    tl_head_df, tl_head_best = None, None
    tl_partial_df, tl_partial_best = None, None
    tl_full_df, tl_full_best = None, None

    # ---- Stage 2.1: Head only
    print("\n" + "-" * 80)
    print("TL Stage 1: Freeze feature extractor, train head only")
    print("-" * 80)

    freeze_feature_extractor(model)

    dl_tl_train_head = DataLoader(
        ds_tl_train,
        batch_size=cfg.tl_head_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    dl_tl_val_head = DataLoader(
        ds_tl_val,
        batch_size=cfg.tl_head_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    model, tl_head_history = fit_model(
        model=model,
        train_loader=dl_tl_train_head,
        val_loader=dl_tl_val_head,
        criterion=criterion,
        epochs=cfg.tl_head_epochs,
        lr=cfg.tl_head_lr,
        weight_decay=cfg.tl_head_weight_decay,
        patience=cfg.tl_head_patience,
        ckpt_path=cfg.tl_head_ckpt,
        stage_name="TL-HEAD",
        target_scaler=target_scaler
    )

    tl_head_df, tl_head_best = summarize_history(
        tl_head_history, stage_name="TL-HEAD"
    )

    # ---- Stage 2.2: Partial unfreeze
    print("\n" + "-" * 80)
    print("TL Stage 2: Unfreeze last feature stage + head")
    print("-" * 80)

    unfreeze_last_feature_stage(model)

    dl_tl_train_partial = DataLoader(
        ds_tl_train,
        batch_size=cfg.tl_partial_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    dl_tl_val_partial = DataLoader(
        ds_tl_val,
        batch_size=cfg.tl_partial_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    model, tl_partial_history = fit_model(
        model=model,
        train_loader=dl_tl_train_partial,
        val_loader=dl_tl_val_partial,
        criterion=criterion,
        epochs=cfg.tl_partial_epochs,
        lr=cfg.tl_partial_lr,
        weight_decay=cfg.tl_partial_weight_decay,
        patience=cfg.tl_partial_patience,
        ckpt_path=cfg.tl_partial_ckpt,
        stage_name="TL-PARTIAL",
        target_scaler=target_scaler
    )

    tl_partial_df, tl_partial_best = summarize_history(
        tl_partial_history, stage_name="TL-PARTIAL"
    )

# ---- Stage 2.3: Full fine-tuning
    if cfg.run_tl_full:
        print("\n" + "-" * 80)
        print("TL Stage 3: Unfreeze full network and fine-tune")
        print("-" * 80)

        unfreeze_all(model)

        dl_tl_train_full = DataLoader(
            ds_tl_train,
            batch_size=cfg.test_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        dl_tl_val_full = DataLoader(
            ds_tl_val,
            batch_size=cfg.tl_full_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

        model, tl_full_history = fit_model(
            model=model,
            train_loader=dl_tl_train_full,
            val_loader=dl_tl_val_full,
            criterion=criterion,
            epochs=cfg.tl_full_epochs,
            lr=cfg.tl_full_lr,
            weight_decay=cfg.tl_full_weight_decay,
            patience=cfg.tl_full_patience,
            ckpt_path=cfg.tl_full_ckpt,
            stage_name="TL-FULL",
            target_scaler=target_scaler
        )

        tl_full_df, tl_full_best = summarize_history(
            tl_full_history, stage_name="TL-FULL"
        )
    else:
        print("\n" + "-" * 80)
        print("TL Stage 3: TL-FULL skipped (cfg.run_tl_full=False)")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("REQUESTED METRICS SUMMARY")
    print("=" * 80)

    print(f"BASE best train RMSE       : {base_best['train_rmse']:.6f}")
    print(f"BASE best val RMSE         : {base_best['val_rmse']:.6f}")
    if 'val_rmse_3d' in base_best.index:
        print(f"BASE best val 3D RMSE      : {base_best['val_rmse_3d']:.6f}")

    print(f"TL-HEAD best train RMSE    : {tl_head_best['train_rmse']:.6f}")
    print(f"TL-HEAD best val RMSE      : {tl_head_best['val_rmse']:.6f}")
    if 'val_rmse_3d' in tl_head_best.index:
        print(
            f"TL-HEAD best val 3D RMSE   : {tl_head_best['val_rmse_3d']:.6f}")

    print(f"TL-PARTIAL best train RMSE : {tl_partial_best['train_rmse']:.6f}")
    print(f"TL-PARTIAL best val RMSE   : {tl_partial_best['val_rmse']:.6f}")
    if 'val_rmse_3d' in tl_partial_best.index:
        print(
            f"TL-PARTIAL best val 3D RMSE: {tl_partial_best['val_rmse_3d']:.6f}")

    if tl_full_best is not None:
        print(f"TL-FULL best train RMSE    : {tl_full_best['train_rmse']:.6f}")
        print(f"TL-FULL best val RMSE      : {tl_full_best['val_rmse']:.6f}")
        if 'val_rmse_3d' in tl_full_best.index:
            print(
                f"TL-FULL best val 3D RMSE   : {tl_full_best['val_rmse_3d']:.6f}")
    else:
        print("TL-FULL best train RMSE    : skipped")
        print("TL-FULL best val RMSE      : skipped")

    # -----------------------------------------------------
    # SELECT BEST TL STAGE AUTOMATICALLY
    # -----------------------------------------------------
    best_tl_name = None
    best_tl_ckpt = None
    best_tl_score = float("inf")

    tl_candidates = []

    if tl_head_best is not None:
        score = float(tl_head_best["val_rmse_3d"]) if "val_rmse_3d" in tl_head_best.index else float(
            tl_head_best["val_rmse"])
        tl_candidates.append(("TL-HEAD", cfg.tl_head_ckpt, score))

    if tl_partial_best is not None:
        score = float(tl_partial_best["val_rmse_3d"]) if "val_rmse_3d" in tl_partial_best.index else float(
            tl_partial_best["val_rmse"])
        tl_candidates.append(("TL-PARTIAL", cfg.tl_partial_ckpt, score))

    if tl_full_best is not None:
        score = float(tl_full_best["val_rmse_3d"]) if "val_rmse_3d" in tl_full_best.index else float(
            tl_full_best["val_rmse"])
        tl_candidates.append(("TL-FULL", cfg.tl_full_ckpt, score))

    for name, ckpt, score in tl_candidates:
        if score < best_tl_score:
            best_tl_score = score
            best_tl_name = name
            best_tl_ckpt = ckpt

    print("\n" + "=" * 80)
    print("BEST TL MODEL SELECTION")
    print("=" * 80)
    print(f"Selected stage : {best_tl_name}")
    print(f"Selected ckpt  : {best_tl_ckpt}")
    print(f"Selected score : {best_tl_score:.6f}")

    # save histories
    if base_df is not None:
        base_df.to_csv("base_training_history.csv", index=False)

    if tl_head_df is not None:
        tl_head_df.to_csv("tl_head_training_history.csv", index=False)

    if tl_partial_df is not None:
        tl_partial_df.to_csv("tl_partial_training_history.csv", index=False)

    if tl_full_df is not None:
        tl_full_df.to_csv("tl_full_training_history.csv", index=False)

    # Reload best TL checkpoint selected from TL stages
    if best_tl_ckpt is not None and os.path.exists(best_tl_ckpt):
        model.load_state_dict(torch.load(
            best_tl_ckpt, map_location=cfg.device))
        print(
            f"\nLoaded best TL checkpoint from: {best_tl_ckpt} ({best_tl_name})")
    else:
        print("\nNo TL checkpoint found for final loading.")

    # -----------------------------------------------------
    # FINAL TEST
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    ds_test = SignalDataset(X_test, y_test_norm, augment=False, is_tl=False)
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.tl_full_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in dl_test:
            x = x.to(cfg.device)
            p = model(x)
            preds.append(p.detach().cpu().numpy())

    y_pred_norm = np.concatenate(preds, axis=0)

    # inverse target normalization for real metrics
    if cfg.use_target_norm:
        y_pred = target_scaler.inverse_transform(y_pred_norm)
    else:
        y_pred = y_pred_norm

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Test MAE  : {test_mae:.6f}")
    print(f"Test RMSE : {test_rmse:.6f}")

    if y_test.shape[1] in [2, 3]:
        test_rmse_3d = rmse_3d(y_test, y_pred)
        print(f"Test 3D RMSE : {test_rmse_3d:.6f}")

    if y_test.shape[1] > 1:
        for d in range(y_test.shape[1]):
            mae_d = mean_absolute_error(y_test[:, d], y_pred[:, d])
            rmse_d = math.sqrt(mean_squared_error(y_test[:, d], y_pred[:, d]))
            print(f"Dim {d}: MAE={mae_d:.6f} | RMSE={rmse_d:.6f}")

    if y_test.shape[1] in [2, 3]:
        euc = np.linalg.norm(y_test - y_pred, axis=1)
        print(f"Mean Euclidean Error : {np.mean(euc):.6f}")
        print(f"Median Euclidean Err : {np.median(euc):.6f}")
        print(f"Std Euclidean Error  : {np.std(euc):.6f}")
        print(f"90th Percentile Err  : {np.percentile(euc, 90):.6f}")
        print(f"95th Percentile Err  : {np.percentile(euc, 95):.6f}")

    np.save("y_test_true.npy", y_test)
    np.save("y_test_pred.npy", y_pred)

    # -----------------------------------------------------
    # SAVE FINAL PREDICTIONS CSV
    # -----------------------------------------------------
    import pandas as pd

    if y_test.shape[1] == 3:
        df_preds = pd.DataFrame({
            "True_X": y_test[:, 0],
            "True_Y": y_test[:, 1],
            "True_Z": y_test[:, 2],
            "Pred_X": y_pred[:, 0],
            "Pred_Y": y_pred[:, 1],
            "Pred_Z": y_pred[:, 2],
        })
    elif y_test.shape[1] == 2:
        df_preds = pd.DataFrame({
            "True_X": y_test[:, 0],
            "True_Y": y_test[:, 1],
            "Pred_X": y_pred[:, 0],
            "Pred_Y": y_pred[:, 1],
        })
    else:
        cols = {}
        for i in range(y_test.shape[1]):
            cols[f"True_{i}"] = y_test[:, i]
            cols[f"Pred_{i}"] = y_pred[:, i]
        df_preds = pd.DataFrame(cols)

    # اضافه کردن خطاهای بعدی
    for i in range(y_test.shape[1]):
        df_preds[f"Err_{i}"] = y_pred[:, i] - y_test[:, i]
        df_preds[f"AbsErr_{i}"] = np.abs(y_pred[:, i] - y_test[:, i])

    # اضافه کردن خطای اقلیدسی اگر خروجی 2 یا 3 بعدی باشد
    if y_test.shape[1] in [2, 3]:
        euc = np.linalg.norm(y_test - y_pred, axis=1)
        df_preds["Euclidean_Error"] = euc

    output_csv = "Final_Predictions.csv"
    df_preds.to_csv(output_csv, index=False)

    print("\nSaved:")
    print("- y_test_true.npy")
    print("- y_test_pred.npy")
    print("- Final_Predictions.csv")
    print(f"- {cfg.base_ckpt}")
    print(f"- {cfg.tl_head_ckpt}")
    print(f"- {cfg.tl_partial_ckpt}")
    if cfg.run_tl_full:
        print(f"- {cfg.tl_full_ckpt}")
    else:
        print("- TL-FULL checkpoint: skipped")
    print(f"- final loaded ckpt: {best_tl_ckpt}")
    print(f"- final loaded stage: {best_tl_name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
