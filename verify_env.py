# =====================================================================
# File: verify_env.py
# Module: THORAX-AI Environment & Hardware Compatibility Diagnostic
# Team: BioByte
# =====================================================================

import sys
import os
from typing import Tuple

def check_python_version() -> bool:
    """Verifies that Python runtime meets the minimum 3.10 requirement."""
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Python Runtime Version: {major}.{minor}.{sys.version_info.micro}")
    if major < 3 or (major == 3 and minor < 10):
        print("    [!] WARNING: Python 3.10+ is recommended for THORAX-AI.")
        return False
    print("    [✓] Python version is verified.")
    return True


def check_compute_device() -> str:
    """
    Identifies active hardware acceleration.
    Checks for AMD DirectML on Windows, falling back to CPU.
    """
    import torch

    device_name = "CPU"
    selected_device = torch.device("cpu")

    # Check for AMD DirectML acceleration
    try:
        import torch_directml
        if torch_directml.is_available():
            selected_device = torch_directml.device()
            device_name = f"AMD DirectML Device ({torch_directml.device_name(0)})"
    except ImportError:
        pass

    # Verify tensor allocation and arithmetic on target device
    test_tensor_a = torch.tensor([1.0, 2.0, 3.0]).to(selected_device)
    test_tensor_b = torch.tensor([4.0, 5.0, 6.0]).to(selected_device)
    result = (test_tensor_a + test_tensor_b).cpu().numpy()

    print(f"[*] Compute Backend: {device_name}")
    print(f"    [✓] PyTorch {torch.__version__} initialized.")
    print(f"    [✓] Mathematical Tensor Operation Verified: {result.tolist()}")
    return device_name


def check_imaging_and_clahe() -> bool:
    """Verifies OpenCV and CLAHE contrast enhancement pipeline."""
    import cv2
    import numpy as np

    # Create dummy 224x224 grayscale image
    dummy_xray = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    # Initialize CLAHE with project parameters: clipLimit=2.0, tileGridSize=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(dummy_xray)

    if enhanced.shape == (224, 224):
        print("[*] OpenCV & CLAHE Contrast Engine:")
        print(f"    [✓] OpenCV Version: {cv2.__version__}")
        print("    [✓] CLAHE (clipLimit=2.0, tileGridSize=(8,8)) verified successfully.")
        return True
    return False


def check_patient_splitter() -> bool:
    """Verifies Scikit-Learn GroupShuffleSplit availability for leak-free splits."""
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np

    dummy_data = np.zeros(10)
    dummy_groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(dummy_data, groups=dummy_groups))
    
    # Ensure disjoint patient groups
    train_patients = set(dummy_groups[train_idx])
    val_patients = set(dummy_groups[val_idx])
    overlap = train_patients.intersection(val_patients)

    if len(overlap) == 0:
        print("[*] Medical Leak-Free Partitioning Engine:")
        print("    [✓] Scikit-Learn GroupShuffleSplit verified with zero group leakage.")
        return True
    return False


def check_directory_structure() -> bool:
    """Verifies that all required project directories are present."""
    required_dirs = ["data", "models", "src"]
    all_exist = True
    print("[*] Project Architecture Structure:")
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"    [✓] Directory found: ./{directory}/")
        else:
            print(f"    [✗] Missing directory: ./{directory}/")
            all_exist = False
    return all_exist


def main():
    print("=" * 65)
    print("THORAX-AI ENVIRONMENT VERIFICATION [TEAM BIOBYTE]")
    print("=" * 65)

    v1 = check_python_version()
    print("-" * 65)
    device_info = check_compute_device()
    print("-" * 65)
    v2 = check_imaging_and_clahe()
    print("-" * 65)
    v3 = check_patient_splitter()
    print("-" * 65)
    v4 = check_directory_structure()
    print("=" * 65)

    if all([v1, v2, v3, v4]):
        print("🎉 SUCCESS: Environment is configured for THORAX-AI development.")
        print(f"   Inference Device: {device_info}")
        print("   Ready to proceed to Step 2 (Data Pipeline).")
    else:
        print("❌ WARNING: One or more components failed verification. Check logs above.")
    print("=" * 65)


if __name__ == "__main__":
    main()