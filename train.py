# =====================================================================
# File: train.py
# Module: THORAX-AI Training, Macro AUC-ROC Validation & Model Export
# Team: BioByte
# Architecture: PyTorch 2.x, DenseNet-121, Scikit-Learn
# =====================================================================

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure root directory is present in sys.path for robust modular imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    TARGET_CLASSES,
    split_cohort_zero_leakage,
    create_clinical_dataloaders,
    ChestXRayDataset
)
from src.model import (
    build_thorax_model_and_criterion,
    compute_positive_class_weights
)


# ---------------------------------------------------------------------
# 1. HARDWARE SELECTION & BACKEND INITIALIZATION
# ---------------------------------------------------------------------
def resolve_compute_device() -> torch.device:
    """
    Selects the optimal execution device.
    Prioritizes CUDA on cloud GPUs, falling back to CPU for local testing.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] Compute Accelerator: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("[*] Compute Accelerator: Standard CPU Engine")
    return device


# ---------------------------------------------------------------------
# 2. VALIDATION & MULTI-LABEL ROC METRICS ENGINE
# ---------------------------------------------------------------------
def evaluate_cohort(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    target_classes: List[str],
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    """
    Runs full cohort validation and computes per-class and Macro AUC-ROC scores.

    Args:
        model (nn.Module): DenseNet-121 model under evaluation.
        val_loader (DataLoader): Validation DataLoader.
        criterion (nn.BCEWithLogitsLoss): Multi-label loss criterion.
        target_classes (List[str]): List of target thoracic diseases.
        device (torch.device): Execution device.

    Returns:
        Tuple[float, float, Dict[str, float]]: (val_loss, macro_auc, per_class_auc_dict)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in val_loader:
            # images: (Batch_Size, 3, 224, 224), labels: (Batch_Size, num_classes)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: obtain unnormalized logits
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Sigmoid activation: map logits to [0.0, 1.0] probabilities
            probabilities = torch.sigmoid(logits)

            all_predictions.append(probabilities.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / max(total_samples, 1)

    # Stack into 2D evaluation matrices: Shape (Total_Val_Samples, num_classes)
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_predictions)

    per_class_auc: Dict[str, float] = {}
    valid_aucs: List[float] = []

    for i, disease in enumerate(target_classes):
        # Calculate AUC-ROC if both positive and negative cases are present
        unique_labels = np.unique(y_true[:, i])
        if len(unique_labels) > 1:
            try:
                score = float(roc_auc_score(y_true[:, i], y_pred[:, i]))
            except ValueError:
                score = 0.50
        else:
            # Clinical baseline fallback when small partition slice lacks positive cases
            score = 0.50

        per_class_auc[disease] = score
        valid_aucs.append(score)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.50
    return epoch_loss, macro_auc, per_class_auc


# ---------------------------------------------------------------------
# 3. TRAINING EPOCH CONTROLLER
# ---------------------------------------------------------------------
def train_single_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Executes a single forward/backward training epoch across all mini-batches.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # images: (Batch_Size, 3, 224, 224), labels: (Batch_Size, num_classes)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward optimization pass
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(total_samples, 1)
    return epoch_loss


# ---------------------------------------------------------------------
# 4. ORCHESTRATION PIPELINE
# ---------------------------------------------------------------------
def run_training_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_classes: List[str],
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    output_model_dir: str = "models",
    checkpoint_name: str = "densenet121_best.pth",
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> float:
    """
    Manages complete training lifecycle, scheduler stepping, and best model serialization.
    """
    device = device or resolve_compute_device()
    os.makedirs(output_model_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(output_model_dir, checkpoint_name)

    # 1. Compute Cost-Sensitive Class Imbalance Weights
    pos_weights = compute_positive_class_weights(train_df, target_classes, device=device)

    # 2. Build DataLoaders
    train_loader, val_loader = create_clinical_dataloaders(
        train_df=train_df,
        val_df=val_df,
        target_conditions=target_classes,
        batch_size=batch_size,
        num_workers=0
    )

    # 3. Build Model Architecture and Loss Criterion
    model, criterion = build_thorax_model_and_criterion(
        num_classes=len(target_classes),
        pos_weights=pos_weights,
        pretrained=pretrained,
        device=device
    )

    # 4. AdamW Optimizer & Plateau Learning Rate Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    best_macro_auc = 0.0

    print("\n" + "=" * 75)
    print("THORAX-AI TRAINING & EVALUATION PIPELINE [TEAM BIOBYTE]")
    print(f"Target Conditions: {', '.join(target_classes)}")
    print(f"Total Epochs: {epochs} | Batch Size: {batch_size} | Base LR: {learning_rate}")
    print("=" * 75 + "\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, macro_auc, class_aucs = evaluate_cohort(model, val_loader, criterion, target_classes, device)

        # Update scheduler based on validation Macro AUC-ROC
        scheduler.step(macro_auc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {macro_auc:.4f}")
        for disease, score in class_aucs.items():
            print(f"   ↳ {disease:<15}: AUC = {score:.4f}")

        # Checkpoint serialization: save only if validation Macro AUC improves
        if macro_auc >= best_macro_auc:
            best_macro_auc = macro_auc
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"   ★ Best model checkpoint saved to: {best_checkpoint_path}")
        print("-" * 75)

    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE. Highest Achieved Validation Macro AUC: {best_macro_auc:.4f}")
    print(f"Deployment Weights Ready at: {best_checkpoint_path}")
    print("=" * 75 + "\n")

    return best_macro_auc


# ---------------------------------------------------------------------
# 5. SYNTHETIC DRY-RUN GENERATOR (FOR LOCAL VS CODE VERIFICATION)
# ---------------------------------------------------------------------
def generate_synthetic_verification_cohort(temp_dir: str = "data/synthetic_train") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates lightweight synthetic radiographs and mock patient metadata
    to verify pipeline execution locally on CPU without downloading the full dataset.
    """
    os.makedirs(temp_dir, exist_ok=True)
    mock_records = []

    for i in range(16):
        img_filename = f"scan_{i:03d}.png"
        img_path = os.path.join(temp_dir, img_filename)

        # Create dummy 224x224 grayscale image
        sample_img = np.random.randint(40, 220, (224, 224), dtype=np.uint8)
        cv2.circle(sample_img, (112, 112), 60, 180, -1)
        cv2.imwrite(img_path, sample_img)

        # Mock multi-label findings across 8 distinct patients (2 scans per patient)
        mock_records.append({
            "Patient ID": f"PATIENT_{(i // 2) + 1:03d}",
            "FilePath": img_path,
            "Cardiomegaly": 1 if i in [0, 1, 4] else 0,
            "Effusion":     1 if i in [2, 3, 5] else 0,
            "Pneumonia":    1 if i in [6, 7] else 0,
            "Pneumothorax": 1 if i in [8, 9] else 0,
            "Atelectasis":  1 if i in [10, 11, 12] else 0
        })

    mock_df = pd.DataFrame(mock_records)
    train_split, val_split = split_cohort_zero_leakage(mock_df, patient_id_column="Patient ID", validation_ratio=0.25, random_seed=42)
    return train_split, val_split


# ---------------------------------------------------------------------
# 6. COMMAND-LINE INTERFACE
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="THORAX-AI Training & Validation Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run a rapid 2-epoch CPU verification with synthetic data")
    parser.add_argument("--csv-path", type=str, default="data/metadata.csv", help="Path to NIH metadata CSV file")
    parser.add_argument("--img-dir", type=str, default="data/images", help="Path to folder containing radiograph images")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    args = parser.parse_args()

    if args.dry_run:
        print("[*] Initiating THORAX-AI local dry-run verification mode...")
        train_df, val_df = generate_synthetic_verification_cohort()
        run_training_pipeline(
            train_df=train_df,
            val_df=val_df,
            target_classes=TARGET_CLASSES,
            epochs=2,
            batch_size=4,
            learning_rate=args.lr,
            checkpoint_name="densenet121_best.pth",
            pretrained=False  # Instant offline initialization for local CPU test
        )
    else:
        if not os.path.exists(args.csv_path):
            print(f"[!] Error: Metadata CSV not found at '{args.csv_path}'.")
            print("    To run an offline dry-run test, use: python train.py --dry-run")
            return

        raw_df = pd.read_csv(args.csv_path)
        train_df, val_df = split_cohort_zero_leakage(raw_df, patient_id_column="Patient ID")
        run_training_pipeline(
            train_df=train_df,
            val_df=val_df,
            target_classes=TARGET_CLASSES,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pretrained=True
        )


if __name__ == "__main__":
    main()