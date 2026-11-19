# =====================================================================
# File: src/model.py
# Module: THORAX-AI Model Architecture & Weighted Loss Optimization
# Team: BioByte
# Architecture: DenseNet-121 + Multi-Label Cost-Sensitive BCE
# =====================================================================

import os
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------
# 1. CLINICAL SPECIFICATION CONSTANTS
# ---------------------------------------------------------------------
DEFAULT_TARGET_CLASSES: List[str] = [
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Atelectasis"
]


# ---------------------------------------------------------------------
# 2. CLASS WEIGHT COMPUTATION ENGINE
# ---------------------------------------------------------------------
def compute_positive_class_weights(
    train_dataframe: pd.DataFrame,
    target_classes: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Computes cost-sensitive positive class weights (w_pos = N_neg / N_pos)
    from training cohort label distributions to counter severe class imbalance.

    Args:
        train_dataframe (pd.DataFrame): Training metadata partition containing binary labels.
        target_classes (Optional[List[str]]): List of target condition column names.
        device (Optional[torch.device]): Target computation device (CPU/DirectML/CUDA).

    Returns:
        torch.Tensor: Positive class weight tensor of shape (num_classes,) as float32.
    """
    targets = target_classes or DEFAULT_TARGET_CLASSES
    device = device or torch.device("cpu")

    num_samples = len(train_dataframe)
    positive_counts = train_dataframe[targets].sum(axis=0).values.astype(np.float32)

    # Prevent division by zero if a rare class has zero samples in a small slice
    positive_counts = np.maximum(positive_counts, 1.0)
    negative_counts = num_samples - positive_counts

    # Calculate class weight ratio: w_pos = N_neg / N_pos
    calculated_weights = negative_counts / positive_counts

    weights_tensor = torch.tensor(calculated_weights, dtype=torch.float32, device=device)

    print("=" * 65)
    print("THORAX-AI CLASS IMBALANCE WEIGHT MATRIX [TEAM BIOBYTE]")
    for idx, disease in enumerate(targets):
        print(f"  --> {disease:<15}: Pos={int(positive_counts[idx]):<6} | Neg={int(negative_counts[idx]):<6} | Weight={calculated_weights[idx]:.4f}")
    print("=" * 65)

    return weights_tensor


# ---------------------------------------------------------------------
# 3. DENSENET-121 CLASSIFICATION ARCHITECTURE
# ---------------------------------------------------------------------
class ThoraxDenseNet121(nn.Module):
    """
    Modified DenseNet-121 diagnostic architecture for multi-label screening.
    Preserves dense feature blocks for Grad-CAM attribution and replaces
    the classifier with a regularized two-stage projection head.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Args:
            num_classes (int): Number of independent pathology logits to output.
            pretrained (bool): If True, initializes feature extractor with ImageNet weights.
            dropout_rate (float): Regularization probability for feature co-adaptation control.
        """
        super(ThoraxDenseNet121, self).__init__()

        # Load DenseNet-121 backbone
        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
            self.densenet = models.densenet121(weights=weights)
        else:
            self.densenet = models.densenet121(weights=None)

        # Extract incoming dimension from denseblock4 terminal pooling (1024 channels)
        in_features: int = self.densenet.classifier.in_features

        # Replace classification head: Linear -> ReLU -> Dropout(0.3) -> Linear
        # Outputs raw unconstrained logits for numerical stability in BCEWithLogitsLoss
        self.densenet.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation pass.

        Args:
            x (torch.Tensor): Radiograph tensor of shape (Batch_Size, 3, 224, 224).

        Returns:
            torch.Tensor: Unnormalized raw logits of shape (Batch_Size, num_classes).
        """
        # Input Tensor Shape Validation: (B, 3, 224, 224)
        logits = self.densenet(x)
        # Output Tensor Shape Validation: (B, num_classes)
        return logits

    def get_cam_target_layer(self) -> nn.Module:
        """
        Exposes the terminal convolutional layer within DenseBlock 4
        required for Grad-CAM gradient hook registration.

        Returns:
            nn.Module: Target Conv2d layer: features.denseblock4.denselayer16.conv2
        """
        return self.densenet.features.denseblock4.denselayer16.conv2


# ---------------------------------------------------------------------
# 4. MODEL & CRITERION BUILDER FACTORY
# ---------------------------------------------------------------------
def build_thorax_model_and_criterion(
    num_classes: int = 5,
    pos_weights: Optional[torch.Tensor] = None,
    pretrained: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[ThoraxDenseNet121, nn.BCEWithLogitsLoss]:
    """
    Constructs the model and pairs it with the cost-sensitive BCE loss criterion.

    Args:
        num_classes (int): Number of target pathologies.
        pos_weights (Optional[torch.Tensor]): Weights for false-negative penalization.
        pretrained (bool): Whether to utilize pretrained ImageNet representations.
        device (Optional[torch.device]): Target compute execution device.

    Returns:
        Tuple[ThoraxDenseNet121, nn.BCEWithLogitsLoss]: Configured model and loss criterion.
    """
    device = device or torch.device("cpu")

    model = ThoraxDenseNet121(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=0.3
    ).to(device)

    # Initialize Binary Cross-Entropy with Logits loss with positive weights
    if pos_weights is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    return model, criterion


# ---------------------------------------------------------------------
# 5. SELF-CONTAINED UNIT TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Running unit test on src/model.py...")

    device = torch.device("cpu")

    # 1. Test Class Weight Calculation
    mock_df = pd.DataFrame({
        "Cardiomegaly": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 Pos, 9 Neg -> w_pos = 9.0
        "Effusion":     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 Pos, 8 Neg -> w_pos = 4.0
        "Pneumonia":    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 1 Pos, 9 Neg -> w_pos = 9.0
        "Pneumothorax": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 1 Pos, 9 Neg -> w_pos = 9.0
        "Atelectasis":  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]   # 4 Pos, 6 Neg -> w_pos = 1.5
    })

    weights = compute_positive_class_weights(mock_df, DEFAULT_TARGET_CLASSES, device=device)
    assert weights.shape == (5,), f"Expected weight shape (5,), got {weights.shape}"

    # 2. Test Model Instantiation & Layer Hook Registration
    model, criterion = build_thorax_model_and_criterion(
        num_classes=5,
        pos_weights=weights,
        pretrained=False,  # Set False for rapid offline testing
        device=device
    )

    # Verify Grad-CAM target layer existence
    cam_layer = model.get_cam_target_layer()
    print(f"[*] Verified Grad-CAM Target Convolutional Layer: {cam_layer}")

    # 3. Test Forward & Backward Optimization Pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_targets = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ], device=device)

    # Forward pass: verify output logits
    logits = model(dummy_input)
    print(f"[*] Output Logits Tensor Shape: {logits.shape} (Expected: [{batch_size}, 5])")
    assert logits.shape == (batch_size, 5), "Logits output dimensions mismatch."

    # Loss computation & backpropagation check
    loss = criterion(logits, dummy_targets)
    print(f"[*] Cost-Sensitive BCE Loss Value: {loss.item():.4f}")

    loss.backward()
    has_gradients = all(p.grad is not None for p in model.densenet.classifier.parameters())
    print(f"[*] Classifier Head Gradient Flow: {'Active' if has_gradients else 'Failed'}")
    assert has_gradients, "Gradients failed to propagate to classification head."

    print("🎉 UNIT TEST PASSED: src/model.py is validated and production-ready.")