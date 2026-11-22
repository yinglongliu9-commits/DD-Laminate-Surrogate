import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MATERIAL_CSV = DATASET_DIR / "material_parameters.csv"
STRENGTH_DIR = DATASET_DIR / "strength"
PNG_DIR = DATASET_DIR / "PNG"
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
DAMAGE_IMAGE_SIZE = 256

# -------------------- Training Config --------------------
# You can edit these defaults directly before running this script.
# Command-line arguments with the same names will override these values.
DEFAULT_CONFIG = {
    "task": "damage",            # strength, damage, or all
    "output_dir": "runs/composite",
    "seed": 42,
    "strength_epochs": 1200,
    "strength_batch_size": 64,
    "strength_lr": 1e-3,
    "damage_epochs": 1000,
    "damage_batch_size": 8,
    "damage_lr": 2e-4,
    "damage_lr_scheduler": True,
    "damage_lr_patience": 40,
    "damage_lr_factor": 0.5,
    "damage_min_lr": 1e-6,
    "damage_base_channels": 48,
    "damage_loss_l1": 1.0,
    "damage_loss_ssim": 0.35,
    "damage_loss_edge": 0.15,
    "export_damage_test_comparisons": True,
    "export_damage_test_error_maps": True,
    "damage_error_alpha_scale": 0.15,
    "weight_decay": 1e-4,
    "patience": 180,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def laminate_angles(a: int, b: int) -> list[float]:
    return [a * 10, -a * 10, b * 10, -b * 10, -b * 10, b * 10, -a * 10, a * 10]


def ab_to_id(a: int, b: int) -> int:
    return b + a * 10 + 1


def load_material_table() -> pd.DataFrame:
    df = pd.read_csv(MATERIAL_CSV)
    df = df.sort_values("material_id").reset_index(drop=True)
    return df


def material_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"material_id", "source_inp", "gangti_density", "gangti_elastic_e", "gangti_poisson"}]


@dataclass
class Standardizer:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean.astype(float).tolist(), std.astype(float).tolist())

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.std, dtype=np.float32) + np.asarray(self.mean, dtype=np.float32)


def build_strength_records() -> tuple[list[dict], list[str]]:
    mat_df = load_material_table()
    mat_cols = material_numeric_columns(mat_df)
    records: list[dict] = []

    for _, mat_row in mat_df.iterrows():
        material_id = mat_row["material_id"]
        cai = pd.read_csv(STRENGTH_DIR / f"{material_id}_CAI.csv").set_index("ID")
        cbi = pd.read_csv(STRENGTH_DIR / f"{material_id}_CBI.csv").set_index("ID")
        material_features = mat_row[mat_cols].astype(float).to_numpy(dtype=np.float32)

        for a in range(10):
            for b in range(10):
                sample_id = ab_to_id(a, b)
                angles = np.asarray(laminate_angles(a, b), dtype=np.float32)
                features = np.concatenate(
                    [
                        material_features,
                        np.asarray([a, b], dtype=np.float32),
                        angles,
                    ]
                )
                records.append(
                    {
                        "material_id": material_id,
                        "a": a,
                        "b": b,
                        "id": sample_id,
                        "features": features,
                        "target": np.asarray(
                            [
                                float(cai.loc[sample_id, "Max_RT"]),
                                float(cbi.loc[sample_id, "Max_RT"]),
                            ],
                            dtype=np.float32,
                        ),
                    }
                )

    feature_names = mat_cols + ["a", "b"] + [f"angle_{i}" for i in range(1, 9)]
    return records, feature_names


class StrengthDataset(Dataset):
    def __init__(self, records: list[dict], x_scaler: Standardizer, y_scaler: Standardizer):
        self.records = records
        x = np.stack([r["features"] for r in records]).astype(np.float32)
        y = np.stack([r["target"] for r in records]).astype(np.float32)
        self.x = x_scaler.transform(x).astype(np.float32)
        self.y = y_scaler.transform(y).astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]), idx


class StrengthMLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def split_indices(n: int, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return indices[:train_end].tolist(), indices[train_end:val_end].tolist(), indices[val_end:].tolist()


def reset_history_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_history_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_strength(model, loader, device, y_scaler: Standardizer):
    model.eval()
    preds, targets, sample_indices = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_loss += F.mse_loss(out, y, reduction="sum").item()
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
            sample_indices.extend(idx.tolist())
    pred = y_scaler.inverse(np.concatenate(preds, axis=0))
    target = y_scaler.inverse(np.concatenate(targets, axis=0))
    mae = np.mean(np.abs(pred - target), axis=0)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))
    denom = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum((target - pred) ** 2, axis=0) / np.maximum(denom, 1e-8)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "pred": pred,
        "target": target,
        "indices": sample_indices,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def train_strength(args, device: torch.device):
    out_dir = Path(args.output_dir) / "strength"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, feature_names = build_strength_records()
    train_idx, val_idx, test_idx = split_indices(len(records), args.seed)
    x_train = np.stack([records[i]["features"] for i in train_idx]).astype(np.float32)
    y_train = np.stack([records[i]["target"] for i in train_idx]).astype(np.float32)
    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)
    dataset = StrengthDataset(records, x_scaler, y_scaler)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = StrengthMLP(len(feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=50, factor=0.5)

    best_val = math.inf
    best_epoch = 0
    history = []
    history_path = out_dir / "training_history.csv"
    reset_history_csv(history_path)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.smooth_l1_loss(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)

        train_loss = total / len(train_loader.dataset)
        val_metrics = evaluate_strength(model, val_loader, device, y_scaler)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        append_history_row(history_path, row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_strength_model.pth")

        elapsed = time.perf_counter() - epoch_start
        print(
            f"[strength] epoch {epoch:04d}/{args.epochs} "
            f"train={train_loss:.6f} val={val_metrics['loss']:.6f} "
            f"MAE(CAI,CBI)=({val_metrics['mae'][0]:.1f},{val_metrics['mae'][1]:.1f}) "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if epoch - best_epoch >= args.patience:
            print(f"[strength] early stop at epoch {epoch}, best epoch {best_epoch}", flush=True)
            break

    model.load_state_dict(torch.load(out_dir / "best_strength_model.pth", map_location=device))
    val_metrics = evaluate_strength(model, val_loader, device, y_scaler)
    test_metrics = evaluate_strength(model, test_loader, device, y_scaler)

    metadata = {
        "task": "strength",
        "feature_names": feature_names,
        "target_names": ["CAI", "CBI"],
        "x_scaler": asdict(x_scaler),
        "y_scaler": asdict(y_scaler),
        "best_epoch": best_epoch,
        "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
        "metrics": {
            "val": {
                "mae": val_metrics["mae"].tolist(),
                "rmse": val_metrics["rmse"].tolist(),
                "r2": val_metrics["r2"].tolist(),
            },
            "test": {
                "mae": test_metrics["mae"].tolist(),
                "rmse": test_metrics["rmse"].tolist(),
                "r2": test_metrics["r2"].tolist(),
            },
        },
    }
    (out_dir / "strength_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(history).to_csv(history_path, index=False)
    save_strength_predictions(out_dir / "test_predictions.csv", records, test_metrics)
    plot_history(out_dir / "training_curve.png", history, "Strength Training")

    print("[strength] best model:", out_dir / "best_strength_model.pth", flush=True)
    print(
        "[strength] test MAE(CAI, CBI): "
        f"{test_metrics['mae'][0]:.2f}, {test_metrics['mae'][1]:.2f}; "
        f"R2: {test_metrics['r2'][0]:.4f}, {test_metrics['r2'][1]:.4f}",
        flush=True,
    )


def save_strength_predictions(path: Path, records: list[dict], metrics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "material_id",
                "a",
                "b",
                "id",
                "cai_true",
                "cbi_true",
                "cai_pred",
                "cbi_pred",
                "cai_abs_error",
                "cbi_abs_error",
            ]
        )
        for local_idx, record_idx in enumerate(metrics["indices"]):
            record = records[record_idx]
            target = metrics["target"][local_idx]
            pred = metrics["pred"][local_idx]
            writer.writerow(
                [
                    record["material_id"],
                    record["a"],
                    record["b"],
                    record["id"],
                    target[0],
                    target[1],
                    pred[0],
                    pred[1],
                    abs(pred[0] - target[0]),
                    abs(pred[1] - target[1]),
                ]
            )


def plot_history(path: Path, history: list[dict], title: str) -> None:
    df = pd.DataFrame(history)
    plt.figure(figsize=(9, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_damage_records() -> tuple[list[dict], list[str]]:
    mat_df = load_material_table()
    mat_cols = material_numeric_columns(mat_df)
    records: list[dict] = []

    for _, mat_row in mat_df.iterrows():
        material_id = mat_row["material_id"]
        material_features = mat_row[mat_cols].astype(float).to_numpy(dtype=np.float32)
        for a in range(10):
            for b in range(10):
                angles = laminate_angles(a, b)
                for layer in range(1, 9):
                    angle = angles[layer - 1]
                    filename = f"{a}_{b}_{layer}_{angle}.png"
                    sdv9 = PNG_DIR / material_id / "SDV9" / filename
                    sdv10 = PNG_DIR / material_id / "SDV10" / filename
                    if not sdv9.exists() or not sdv10.exists():
                        raise FileNotFoundError(f"missing damage target: {sdv9} or {sdv10}")
                    features = np.concatenate(
                        [
                            material_features,
                            np.asarray([a, b], dtype=np.float32),
                            np.asarray(angles, dtype=np.float32),
                            np.asarray([layer, angle], dtype=np.float32),
                        ]
                    )
                    records.append(
                        {
                            "material_id": material_id,
                            "a": a,
                            "b": b,
                            "layer": layer,
                            "angle": angle,
                            "features": features,
                            "sdv9": sdv9,
                            "sdv10": sdv10,
                        }
                    )

    feature_names = mat_cols + ["a", "b"] + [f"angle_{i}" for i in range(1, 9)] + ["layer", "layer_angle"]
    return records, feature_names


class DamageDataset(Dataset):
    def __init__(self, records: list[dict], x_scaler: Standardizer, image_size: int):
        self.records = records
        x = np.stack([r["features"] for r in records]).astype(np.float32)
        self.x = x_scaler.transform(x).astype(np.float32)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        target = np.concatenate(
            [
                self._load_rgb(rec["sdv9"]),
                self._load_rgb(rec["sdv10"]),
            ],
            axis=0,
        )
        return torch.from_numpy(self.x[idx]), torch.from_numpy(target), idx

    def _load_rgb(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), RESAMPLE_BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        groups = min(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.film1 = FiLM(cond_dim, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.film2 = FiLM(cond_dim, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, cond)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, cond)
        return self.act(x + residual)


class FiLMUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.block1 = FiLMResBlock(out_channels, cond_dim)
        self.block2 = FiLMResBlock(out_channels, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.block1(x, cond)
        return self.block2(x, cond)


class DamageGenerator(nn.Module):
    def __init__(self, in_features: int, base_channels: int = 48, out_channels: int = 6):
        super().__init__()
        cond_dim = 256
        c16 = base_channels * 4
        c32 = base_channels * 4
        c64 = base_channels * 2
        c128 = base_channels
        c256 = max(base_channels // 2, 16)

        self.condition = nn.Sequential(
            nn.Linear(in_features, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Linear(cond_dim, c16 * 16 * 16),
            nn.GELU(),
        )
        self.c16 = c16
        self.stem = FiLMResBlock(c16, cond_dim)
        self.up32 = FiLMUpBlock(c16, c32, cond_dim)
        self.up64 = FiLMUpBlock(c32, c64, cond_dim)
        self.up128 = FiLMUpBlock(c64, c128, cond_dim)
        self.up256 = FiLMUpBlock(c128, c256, cond_dim)
        self.head = nn.Sequential(
            nn.Conv2d(c256, c256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c256, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cond = self.condition(x)
        x = self.project(cond)
        x = x.view(x.shape[0], self.c16, 16, 16)
        x = self.stem(x, cond)
        x = self.up32(x, cond)
        x = self.up64(x, cond)
        x = self.up128(x, cond)
        x = self.up256(x, cond)
        return self.head(x)


def train_damage(args, device: torch.device):
    out_dir = Path(args.output_dir) / "damage"
    out_dir.mkdir(parents=True, exist_ok=True)
    records, feature_names = build_damage_records()
    train_idx, val_idx, test_idx = split_indices(len(records), args.seed)
    x_train = np.stack([records[i]["features"] for i in train_idx]).astype(np.float32)
    x_scaler = Standardizer.fit(x_train)

    dataset = DamageDataset(records, x_scaler, DAMAGE_IMAGE_SIZE)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.damage_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.damage_batch_size, shuffle=False, num_workers=0)

    model = DamageGenerator(len(feature_names), base_channels=args.damage_base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.damage_lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.damage_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=args.damage_lr_patience,
            factor=args.damage_lr_factor,
            min_lr=args.damage_min_lr,
        )
    best_val = math.inf
    history = []
    history_path = out_dir / "training_history.csv"
    reset_history_csv(history_path)

    for epoch in range(1, args.damage_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        train_l1_total = 0.0
        train_ssim_total = 0.0
        train_edge_total = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            if out.shape != y.shape:
                raise ValueError(f"damage output shape {tuple(out.shape)} does not match target {tuple(y.shape)}")
            loss_parts = damage_loss_components(out, y, args)
            loss = loss_parts["loss"]
            loss.backward()
            optimizer.step()
            batch_size = x.size(0)
            total += loss.item() * batch_size
            train_l1_total += loss_parts["l1"].item() * batch_size
            train_ssim_total += loss_parts["ssim"].item() * batch_size
            train_edge_total += loss_parts["edge"].item() * batch_size

        train_loss = total / len(train_loader.dataset)
        train_l1 = train_l1_total / len(train_loader.dataset)
        train_ssim = train_ssim_total / len(train_loader.dataset)
        train_edge = train_edge_total / len(train_loader.dataset)
        val_metrics = evaluate_damage_metrics(model, val_loader, device, args)
        val_loss = val_metrics["loss"]
        old_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l1_loss": train_l1,
            "train_ssim_loss": train_ssim,
            "train_edge_loss": train_edge,
            "val_loss": val_loss,
            "val_l1_loss": val_metrics["l1"],
            "val_ssim_loss": val_metrics["ssim"],
            "val_edge_loss": val_metrics["edge"],
            "lr": current_lr,
        }
        history.append(row)
        append_history_row(history_path, row)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best_damage_model.pth")
            save_damage_preview(model, dataset, val_idx[:4], device, out_dir / "preview.png")
        elapsed = time.perf_counter() - epoch_start
        print(
            f"[damage] epoch {epoch:04d}/{args.damage_epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"train_parts=(l1:{train_l1:.6f},ssim:{train_ssim:.6f},edge:{train_edge:.6f}) "
            f"val_parts=(l1:{val_metrics['l1']:.6f},ssim:{val_metrics['ssim']:.6f},edge:{val_metrics['edge']:.6f}) "
            f"lr={current_lr:.2e} time={elapsed:.1f}s",
            flush=True,
        )
        if current_lr < old_lr:
            print(f"[damage] learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}", flush=True)

    metadata = {
        "task": "damage",
        "feature_names": feature_names,
        "target_channels": ["SDV9_R", "SDV9_G", "SDV9_B", "SDV10_R", "SDV10_G", "SDV10_B"],
        "image_size": DAMAGE_IMAGE_SIZE,
        "architecture": "FiLM-conditioned 256 decoder",
        "loss": {
            "l1": args.damage_loss_l1,
            "ssim": args.damage_loss_ssim,
            "edge": args.damage_loss_edge,
        },
        "lr_scheduler": {
            "enabled": args.damage_lr_scheduler,
            "type": "ReduceLROnPlateau",
            "patience": args.damage_lr_patience,
            "factor": args.damage_lr_factor,
            "min_lr": args.damage_min_lr,
        },
        "x_scaler": asdict(x_scaler),
        "splits": {"train": train_idx, "val": val_idx, "test": test_idx},
    }
    (out_dir / "damage_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(history_path, index=False)
    plot_history(out_dir / "training_curve.png", history, "Damage Training")
    print("[damage] best model:", out_dir / "best_damage_model.pth", flush=True)

    if args.export_damage_test_comparisons:
        model.load_state_dict(torch.load(out_dir / "best_damage_model.pth", map_location=device))
        save_damage_comparisons(
            model=model,
            dataset=dataset,
            indices=test_idx,
            device=device,
            output_dir=out_dir / "comparasion" / "test",
            batch_size=args.damage_batch_size,
            split_name="test",
        )
        if args.export_damage_test_error_maps:
            save_damage_error_maps(
                model=model,
                dataset=dataset,
                indices=test_idx,
                device=device,
                output_dir=out_dir / "error_map" / "test",
                batch_size=args.damage_batch_size,
                alpha_scale=args.damage_error_alpha_scale,
                split_name="test",
            )


def evaluate_damage_loss(model, loader, device: torch.device, args) -> float:
    return evaluate_damage_metrics(model, loader, device, args)["loss"]


def evaluate_damage_metrics(model, loader, device: torch.device, args) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "edge": 0.0}
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if out.shape != y.shape:
                raise ValueError(f"damage output shape {tuple(out.shape)} does not match target {tuple(y.shape)}")
            loss_parts = damage_loss_components(out, y, args)
            batch_size = x.size(0)
            for name, value in loss_parts.items():
                totals[name] += value.item() * batch_size
    n = max(len(loader.dataset), 1)
    return {name: value / n for name, value in totals.items()}


def damage_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    return damage_loss_components(pred, target, args)["loss"]


def damage_loss_components(pred: torch.Tensor, target: torch.Tensor, args) -> dict[str, torch.Tensor]:
    if args is None:
        l1_weight = DEFAULT_CONFIG["damage_loss_l1"]
        ssim_weight = DEFAULT_CONFIG["damage_loss_ssim"]
        edge_weight = DEFAULT_CONFIG["damage_loss_edge"]
    else:
        l1_weight = args.damage_loss_l1
        ssim_weight = args.damage_loss_ssim
        edge_weight = args.damage_loss_edge

    l1 = F.l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    edge = edge_loss(pred, target)
    loss = l1_weight * l1 + ssim_weight * ssim + edge_weight * edge
    return {"loss": loss, "l1": l1, "ssim": ssim, "edge": edge}


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    padding = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred * pred, window_size, stride=1, padding=padding) - mu_pred_sq
    sigma_target = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_target_sq
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=padding) - mu_cross

    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    ssim = numerator / denominator.clamp_min(1e-8)
    return 1.0 - ssim.clamp(0.0, 1.0).mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def save_damage_preview(model, dataset: DamageDataset, indices: list[int], device: torch.device, path: Path) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in indices:
            x, target, _ = dataset[idx]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()
            if pred.shape != target.shape:
                raise ValueError(f"preview shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")
            rows.append(make_preview_row(target, pred))
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(canvas).save(path)


def save_damage_comparisons(
    model: nn.Module,
    dataset: DamageDataset,
    indices: list[int],
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    split_name: str = "test",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    saved = 0
    total = len(indices)

    with torch.no_grad():
        for x, target, sample_indices in loader:
            pred = model(x.to(device)).cpu()
            if pred.shape != target.shape:
                raise ValueError(f"comparison shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")

            for batch_idx, sample_idx in enumerate(sample_indices.tolist()):
                record = dataset.records[sample_idx]
                comparison = make_preview_row(target[batch_idx], pred[batch_idx])
                Image.fromarray(comparison).save(output_dir / damage_comparison_name(record))
                saved += 1

            if saved == total or saved % 100 == 0:
                print(f"[damage] saved {split_name} comparisons {saved}/{total}", flush=True)

    print(f"[damage] {split_name} comparisons saved to: {output_dir}", flush=True)


def save_damage_error_maps(
    model: nn.Module,
    dataset: DamageDataset,
    indices: list[int],
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    alpha_scale: float,
    split_name: str = "test",
) -> None:
    sdv9_dir = output_dir / "SDV9"
    sdv10_dir = output_dir / "SDV10"
    comparison_dir = output_dir / "comparison"
    for path in (sdv9_dir, sdv10_dir, comparison_dir):
        path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    saved = 0
    total = len(indices)

    with torch.no_grad():
        for x, target, sample_indices in loader:
            pred = model(x.to(device)).cpu()
            if pred.shape != target.shape:
                raise ValueError(f"error-map shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")

            for batch_idx, sample_idx in enumerate(sample_indices.tolist()):
                record = dataset.records[sample_idx]
                filename = damage_comparison_name(record)
                sdv9_error = make_transparent_error_map(target[batch_idx, 0:3], pred[batch_idx, 0:3], alpha_scale)
                sdv10_error = make_transparent_error_map(target[batch_idx, 3:6], pred[batch_idx, 3:6], alpha_scale)

                Image.fromarray(sdv9_error, mode="RGBA").save(sdv9_dir / filename)
                Image.fromarray(sdv10_error, mode="RGBA").save(sdv10_dir / filename)

                comparison = make_error_comparison_row(
                    target[batch_idx],
                    pred[batch_idx],
                    sdv9_error,
                    sdv10_error,
                )
                Image.fromarray(comparison, mode="RGBA").save(comparison_dir / filename)
                saved += 1

            if saved == total or saved % 100 == 0:
                print(f"[damage] saved {split_name} error maps {saved}/{total}", flush=True)

    print(f"[damage] {split_name} error maps saved to: {output_dir}", flush=True)


def make_transparent_error_map(
    target: torch.Tensor,
    pred: torch.Tensor,
    alpha_scale: float,
) -> np.ndarray:
    error = (target - pred).abs().mean(dim=0).detach().numpy()
    t = np.clip(error / max(alpha_scale, 1e-8), 0.0, 1.0)

    light_blue = np.array([204, 229, 255], dtype=np.float32)
    dark_blue = np.array([0, 24, 96], dtype=np.float32)
    rgb = light_blue * (1.0 - t[..., None]) + dark_blue * t[..., None]

    rgba = np.zeros((*t.shape, 4), dtype=np.uint8)
    rgba[..., 0:3] = rgb.astype(np.uint8)
    rgba[..., 3] = (t * 255).astype(np.uint8)
    return rgba


def make_error_comparison_row(
    target: torch.Tensor,
    pred: torch.Tensor,
    sdv9_error: np.ndarray,
    sdv10_error: np.ndarray,
) -> np.ndarray:
    # Row layout: SDV9 true | SDV9 pred | SDV9 error | SDV10 true | SDV10 pred | SDV10 error
    blocks = [
        to_rgba(target[0:3]),
        to_rgba(pred[0:3]),
        composite_rgba_on_white(sdv9_error),
        to_rgba(target[3:6]),
        to_rgba(pred[3:6]),
        composite_rgba_on_white(sdv10_error),
    ]
    gap = np.full((blocks[0].shape[0], 6, 4), 255, dtype=np.uint8)
    return np.concatenate([blocks[0], gap, blocks[1], gap, blocks[2], gap, blocks[3], gap, blocks[4], gap, blocks[5]], axis=1)


def composite_rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    white = np.full_like(rgb, 255.0)
    composited = rgb * alpha + white * (1.0 - alpha)
    opaque_alpha = np.full((*rgba.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([composited.astype(np.uint8), opaque_alpha], axis=2)


def damage_comparison_name(record: dict) -> str:
    angle = record["angle"]
    if float(angle).is_integer():
        angle_text = str(int(angle))
    else:
        angle_text = str(angle).replace(".", "p")
    return f"{record['material_id']}_{record['a']}_{record['b']}_layer{record['layer']}_angle{angle_text}.png"


def make_preview_row(target: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    # Row layout: SDV9 true | SDV9 pred | SDV10 true | SDV10 pred
    blocks = []
    for start in (0, 3):
        blocks.append(to_uint8(target[start : start + 3]))
        blocks.append(to_uint8(pred[start : start + 3]))
    gap = np.full((blocks[0].shape[0], 6, 3), 255, dtype=np.uint8)
    return np.concatenate([blocks[0], gap, blocks[1], gap, blocks[2], gap, blocks[3]], axis=1)


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().clamp(0, 1).numpy().transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)


def to_rgba(tensor: torch.Tensor) -> np.ndarray:
    rgb = to_uint8(tensor)
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train composite laminate strength and damage models.")
    parser.add_argument("--task", choices=["strength", "damage", "all"], default=DEFAULT_CONFIG["task"])
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["strength_batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["strength_epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["strength_lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--damage-epochs", type=int, default=DEFAULT_CONFIG["damage_epochs"])
    parser.add_argument("--damage-batch-size", type=int, default=DEFAULT_CONFIG["damage_batch_size"])
    parser.add_argument("--damage-lr", type=float, default=DEFAULT_CONFIG["damage_lr"])
    parser.add_argument(
        "--damage-lr-scheduler",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["damage_lr_scheduler"],
    )
    parser.add_argument("--damage-lr-patience", type=int, default=DEFAULT_CONFIG["damage_lr_patience"])
    parser.add_argument("--damage-lr-factor", type=float, default=DEFAULT_CONFIG["damage_lr_factor"])
    parser.add_argument("--damage-min-lr", type=float, default=DEFAULT_CONFIG["damage_min_lr"])
    parser.add_argument("--damage-base-channels", type=int, default=DEFAULT_CONFIG["damage_base_channels"])
    parser.add_argument("--damage-loss-l1", type=float, default=DEFAULT_CONFIG["damage_loss_l1"])
    parser.add_argument("--damage-loss-ssim", type=float, default=DEFAULT_CONFIG["damage_loss_ssim"])
    parser.add_argument("--damage-loss-edge", type=float, default=DEFAULT_CONFIG["damage_loss_edge"])
    parser.add_argument(
        "--export-damage-test-comparisons",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["export_damage_test_comparisons"],
    )
    parser.add_argument(
        "--export-damage-test-error-maps",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["export_damage_test_error_maps"],
    )
    parser.add_argument("--damage-error-alpha-scale", type=float, default=DEFAULT_CONFIG["damage_error_alpha_scale"])
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    print(
        "config: "
        f"task={args.task}, strength_epochs={args.epochs}, strength_lr={args.lr}, "
        f"strength_batch_size={args.batch_size}, damage_epochs={args.damage_epochs}, "
        f"damage_lr={args.damage_lr}, damage_batch_size={args.damage_batch_size}, "
        f"damage_lr_scheduler={args.damage_lr_scheduler}, damage_lr_patience={args.damage_lr_patience}, "
        f"damage_lr_factor={args.damage_lr_factor}, damage_min_lr={args.damage_min_lr}, "
        f"damage_base_channels={args.damage_base_channels}, "
        f"damage_loss=(l1:{args.damage_loss_l1}, ssim:{args.damage_loss_ssim}, edge:{args.damage_loss_edge}), "
        f"export_damage_test_comparisons={args.export_damage_test_comparisons}, "
        f"export_damage_test_error_maps={args.export_damage_test_error_maps}",
        flush=True,
    )
    if args.task in {"strength", "all"}:
        train_strength(args, device)
    if args.task in {"damage", "all"}:
        train_damage(args, device)


if __name__ == "__main__":
    main()
