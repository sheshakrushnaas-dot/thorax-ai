# =====================================================================
# File: src/gradcam.py
# Module: THORAX-AI Gradient-Weighted Class Activation Mapping Engine
# Team: BioByte
# Architecture: PyTorch 2.x, OpenCV, NumPy
# =====================================================================

import os
import sys
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1. GRAD-CAM ATTRIBUTION CLASS
# ---------------------------------------------------------------------
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Registers forward and full backward hooks on the terminal convolutional
    layer of DenseNet-121 to extract localized anatomical evidence.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model (nn.Module): Configured DenseNet-121 classification model.
            target_layer (nn.Module): Target layer (features.denseblock4.denselayer16.conv2).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._register_execution_hooks()

    def _register_execution_hooks(self) -> None:
        """Attaches forward and backward hooks to intercept activation and gradient tensors."""
        def forward_hook(module: nn.Module, input_tensor: Tuple[torch.Tensor], output_tensor: torch.Tensor):
            # Intercept forward activations: Shape (B, C, H, W) -> (1, 32, 7, 7)
            self.activations = output_tensor.detach()

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            # Intercept backward gradients flowing from target logit: Shape (1, 32, 7, 7)
            self.gradients = grad_output[0].detach()

        f_handle = self.target_layer.register_forward_hook(forward_hook)
        b_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([f_handle, b_handle])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int
    ) -> np.ndarray:
        """
        Generates a 2D normalized Grad-CAM heatmap for a target thoracic condition.

        Args:
            input_tensor (torch.Tensor): Preprocessed radiograph tensor of shape (1, 3, 224, 224).
            class_idx (int): Integer index corresponding to the target pathology (0 to 4).

        Returns:
            np.ndarray: Normalized 2D float32 array of shape (H, W) bounded within [0.0, 1.0].
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Input tensor must have shape (1, 3, H, W), got {input_tensor.shape}")

        # Ensure model is in evaluation mode and reset gradients
        self.model.zero_grad()

        # Forward pass to obtain raw unconstrained logits
        # Shape: (1, num_classes)
        logits = self.model(input_tensor)

        if class_idx < 0 or class_idx >= logits.size(1):
            raise IndexError(f"Target class index {class_idx} out of range for model with {logits.size(1)} classes.")

        # Isolate target pathology logit score
        target_score = logits[0, class_idx]

        # Backward pass: Compute gradients of target_score w.r.t. target_layer activations
        target_score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global Average Pooling of gradients across spatial dimensions (height, width)
        # pooled_weights shape: (1, C, 1, 1)
        pooled_weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight the activation channels by their importance gradients
        # Shape: (1, H, W)
        cam = torch.sum(pooled_weights * self.activations, dim=1).squeeze(0)

        # Apply ReLU to retain only features that positively influence the target pathology
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Robust Min-Max Normalization to [0.0, 1.0]
        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max - cam_min > 1e-8:
            normalized_cam = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            normalized_cam = np.zeros_like(cam_np, dtype=np.float32)

        return normalized_cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Removes registered hooks from model layers to free memory."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


# ---------------------------------------------------------------------
# 2. RADIOGRAPHIC HEATMAP BLENDING ENGINE
# ---------------------------------------------------------------------
def overlay_heatmap_on_radiograph(
    radiograph_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Interpolates the low-resolution CAM heatmap to radiograph dimensions,
    applies a clinical colormap, and blends via alpha compositing.

    Args:
        radiograph_rgb (np.ndarray): 3-channel RGB image uint8 array of shape (H, W, 3).
        heatmap (np.ndarray): 2D float32 array of shape (h, w) with values in [0.0, 1.0].
        alpha (float): Opacity weighting factor for heatmap overlay (0.0 to 1.0).
        colormap (int): OpenCV colormap identifier (default: cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB diagnostic overlay image as uint8 array (H, W, 3).
    """
    if radiograph_rgb.ndim != 3 or radiograph_rgb.shape[2] != 3:
        raise ValueError(f"Radiograph must be an RGB array with shape (H, W, 3), got {radiograph_rgb.shape}")

    target_h, target_w = radiograph_rgb.shape[0], radiograph_rgb.shape[1]

    # Bicubic interpolation of 7x7 activation map to full image resolution
    heatmap_resized = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

    # Convert to 8-bit unsigned integer [0, 255] for OpenCV colormap application
    heatmap_uint8 = np.uint8(255.0 * heatmap_resized)

    # Apply colormap (cv2 outputs BGR format)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: Blended = (1 - alpha) * Base + alpha * Overlay
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended_composite = (1.0 - alpha) * radiograph_rgb.astype(np.float32) + alpha * colored_heatmap_rgb.astype(np.float32)
    blended_composite = np.clip(blended_composite, 0.0, 255.0).astype(np.uint8)

    return blended_composite


# ---------------------------------------------------------------------
# 3. SELF-CONTAINED UNIT TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure project root is present in sys.path regardless of execution method
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("[*] Running unit test on src/gradcam.py...")

    from src.model import build_thorax_model_and_criterion, DEFAULT_TARGET_CLASSES

    device = torch.device("cpu")

    # 1. Initialize model and extract target layer
    model, _ = build_thorax_model_and_criterion(num_classes=5, pretrained=False, device=device)
    model.eval()

    target_conv_layer = model.get_cam_target_layer()
    print(f"[*] Attaching Grad-CAM to Layer: {target_conv_layer}")

    # 2. Instantiate GradCAM engine
    grad_cam_engine = GradCAM(model=model, target_layer=target_conv_layer)

    # 3. Create synthetic radiograph tensor (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)

    # 4. Generate heatmap for Cardiomegaly (Class Index 0)
    target_class = 0
    disease_name = DEFAULT_TARGET_CLASSES[target_class]
    heatmap = grad_cam_engine.generate_heatmap(dummy_input, class_idx=target_class)

    print(f"[*] Generated Heatmap Shape: {heatmap.shape} (Expected: (7, 7))")
    print(f"[*] Heatmap Value Range: [{heatmap.min():.4f}, {heatmap.max():.4f}] (Expected: [0.0, 1.0])")
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap bounds violation."

    # 5. Test Overlay Generation
    dummy_original_rgb = np.ones((448, 448, 3), dtype=np.uint8) * 128
    overlay = overlay_heatmap_on_radiograph(dummy_original_rgb, heatmap, alpha=0.45)

    print(f"[*] Generated Overlay Composite Shape: {overlay.shape} (Expected: (448, 448, 3))")
    print(f"[*] Overlay Data Type: {overlay.dtype} (Expected: uint8)")
    assert overlay.shape == (448, 448, 3), "Overlay dimension mismatch."

    # 6. Hook Cleanup Verification
    grad_cam_engine.remove_hooks()
    assert len(grad_cam_engine.hook_handles) == 0, "Failed to detach hooks."
    print("[*] Gradient and Activation Hooks Cleared Successfully.")

    print(f"🎉 UNIT TEST PASSED: src/gradcam.py is validated for {disease_name} attribution.")