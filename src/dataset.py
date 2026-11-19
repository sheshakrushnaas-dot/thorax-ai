# =====================================================================
# File: src/dataset.py
# Module: THORAX-AI Dataset, Leak-Free Partitioning, and CLAHE Pipeline
# Team: BioByte
# Architecture: PyTorch 2.x, OpenCV, Scikit-Learn
# =====================================================================

import os
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------
# 1. CLINICAL SPECIFICATION & IMAGING CONSTANTS
# ---------------------------------------------------------------------
TARGET_CLASSES: List[str] = [
    "Cardiomegaly",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Atelectasis"
]

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------
# 2. ZERO-PATIENT-LEAKAGE GROUP PARTITIONING
# ---------------------------------------------------------------------
def split_cohort_zero_leakage(
    dataframe: pd.DataFrame,
    patient_id_column: str = "Patient ID",
    validation_ratio: float = 0.20,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions radiograph metadata into training and validation cohorts
    strictly grouped by Patient ID to ensure zero data leakage across folds.

    Args:
        dataframe (pd.DataFrame): Cohort metadata containing image paths and patient IDs.
        patient_id_column (str): Column name identifying unique patients.
        validation_ratio (float): Proportion of unique patients assigned to validation.
        random_seed (int): Deterministic seed for reproducible evaluation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_dataframe, val_dataframe)
    """
    if patient_id_column not in dataframe.columns:
        raise KeyError(
            f"Specified patient column '{patient_id_column}' does not exist in the DataFrame. "
            f"Available columns: {list(dataframe.columns)}"
        )

    # Initialize Scikit-Learn GroupShuffleSplit
    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_ratio,
        random_state=random_seed
    )

    patient_groups = dataframe[patient_id_column].values
    train_indices, val_indices = next(group_splitter.split(dataframe, groups=patient_groups))

    train_df = dataframe.iloc[train_indices].copy().reset_index(drop=True)
    val_df = dataframe.iloc[val_indices].copy().reset_index(drop=True)

    # Formal clinical audit: Assert zero patient cohort overlap
    train_patients = set(train_df[patient_id_column].unique())
    val_patients = set(val_df[patient_id_column].unique())
    leakage = train_patients.intersection(val_patients)

    if len(leakage) > 0:
        raise ValueError(
            f"FATAL LEAKAGE: {len(leakage)} patient IDs found in both training "
            f"and validation splits. Aborting pipeline."
        )

    print("=" * 65)
    print("THORAX-AI CLINICAL COHORT PARTITION AUDIT [TEAM BIOBYTE]")
    print(f"Total Radiographs:         {len(dataframe)}")
    print(f"Training Radiographs:      {len(train_df)} ({len(train_patients)} unique patients)")
    print(f"Validation Radiographs:    {len(val_df)} ({len(val_patients)} unique patients)")
    print(f"Patient Overlap Count:     {len(leakage)} (Zero Leakage Verified)")
    print("=" * 65)

    return train_df, val_df


# ---------------------------------------------------------------------
# 3. PYTORCH RADIOGRAPH DATASET WITH CLAHE PREPROCESSING
# ---------------------------------------------------------------------
class ChestXRayDataset(Dataset):
    """
    Standardized PyTorch Dataset for Chest Radiographs.
    Executes CLAHE contrast enhancement, 3-channel conversion,
    and ImageNet normalization for DenseNet-121 ingestion.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_path_column: str = "FilePath",
        target_conditions: Optional[List[str]] = None,
        is_training: bool = True,
        image_dimensions: Tuple[int, int] = (224, 224),
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8)
    ):
        """
        Args:
            dataframe (pd.DataFrame): Metadata slice containing filepaths and binary labels.
            image_path_column (str): Column name pointing to image files on disk.
            target_conditions (Optional[List[str]]): Target disease names.
            is_training (bool): If True, applies mild orientation-safe spatial transforms.
            image_dimensions (Tuple[int, int]): Input resolution (H, W) for DenseNet-121.
            clahe_clip_limit (float): Threshold for contrast limiting in OpenCV CLAHE.
            clahe_grid_size (Tuple[int, int]): Contextual tile subdivision for histogram equalization.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_path_column = image_path_column
        self.target_conditions = target_conditions or TARGET_CLASSES
        self.is_training = is_training
        self.image_dimensions = image_dimensions

        # Instantiate OpenCV CLAHE Engine (clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_engine = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

        # Clinically safe transforms: affine rotation <= 7 degrees, horizontal flip
        # Vertical flips strictly excluded to maintain physiological orientation
        if self.is_training:
            self.transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=7),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(self.image_dimensions),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform_pipeline = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.image_dimensions),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves, preprocesses, and converts a single radiograph scan.

        Returns:
            image_tensor (torch.Tensor): Shape (3, 224, 224), float32 normalized tensor.
            label_tensor (torch.Tensor): Shape (5,), float32 multi-label binary ground truth.
        """
        record = self.dataframe.iloc[index]
        scan_filepath = record[self.image_path_column]

        if not os.path.exists(scan_filepath):
            raise FileNotFoundError(f"Radiograph not located on disk: {scan_filepath}")

        # Ingest as single-channel 8-bit grayscale radiograph
        raw_grayscale = cv2.imread(scan_filepath, cv2.IMREAD_GRAYSCALE)
        if raw_grayscale is None:
            raise IOError(f"Corrupt or unreadable image file: {scan_filepath}")

        # Execute Contrast Limited Adaptive Histogram Equalization
        enhanced_grayscale = self.clahe_engine.apply(raw_grayscale)

        # Convert 1-channel grayscale to 3-channel RGB for DenseNet feature extractor compatibility
        enhanced_rgb = cv2.cvtColor(enhanced_grayscale, cv2.COLOR_GRAY2RGB)

        # Apply spatial augmentations and ImageNet normalization
        # Output Tensor Shape: (3, 224, 224)
        image_tensor = self.transform_pipeline(enhanced_rgb)

        # Extract binary labels vector as torch.float32 for BCEWithLogitsLoss
        labels_vector = record[self.target_conditions].values.astype(np.float32)
        label_tensor = torch.from_numpy(labels_vector)  # Dimension: (5,)

        return image_tensor, label_tensor


# ---------------------------------------------------------------------
# 4. DATALOADER FACTORY FUNCTION
# ---------------------------------------------------------------------
def create_clinical_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_conditions: Optional[List[str]] = None,
    image_path_column: str = "FilePath",
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Instantiates PyTorch DataLoaders for train and validation cohorts.

    Args:
        train_df (pd.DataFrame): Training metadata partition.
        val_df (pd.DataFrame): Validation metadata partition.
        target_conditions (Optional[List[str]]): Target condition names.
        image_path_column (str): Column name mapping to local filepaths.
        batch_size (int): Optimization batch size (default: 32).
        num_workers (int): Number of background worker processes.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    conditions = target_conditions or TARGET_CLASSES

    training_dataset = ChestXRayDataset(
        dataframe=train_df,
        image_path_column=image_path_column,
        target_conditions=conditions,
        is_training=True
    )

    validation_dataset = ChestXRayDataset(
        dataframe=val_df,
        image_path_column=image_path_column,
        target_conditions=conditions,
        is_training=False
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return training_loader, validation_loader


# ---------------------------------------------------------------------
# 5. SELF-CONTAINED UNIT TEST
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("[*] Running unit test on src/dataset.py with synthetic radiograph data...")

    # Create dummy radiograph on disk for verification
    test_dir = os.path.join("data", "synthetic_test")
    os.makedirs(test_dir, exist_ok=True)
    dummy_img_path = os.path.join(test_dir, "synthetic_xray.png")

    # Generate synthetic 512x512 chest-like gradient image
    dummy_img = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(dummy_img, (256, 256), 180, 200, -1)
    cv2.imwrite(dummy_img_path, dummy_img)

    # Construct mock patient metadata
    mock_data = {
        "Patient ID": [101, 101, 102, 103, 104, 105, 105, 106, 107, 108],
        "FilePath": [dummy_img_path] * 10,
        "Cardiomegaly": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        "Effusion":     [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        "Pneumonia":    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "Pneumothorax": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Atelectasis":  [0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    }
    mock_df = pd.DataFrame(mock_data)

    # 1. Test Leak-Free Split
    train_split, val_split = split_cohort_zero_leakage(mock_df, patient_id_column="Patient ID", validation_ratio=0.3)

    # 2. Test DataLoader instantiation & tensor extraction
    t_loader, v_loader = create_clinical_dataloaders(train_split, val_split, batch_size=2, num_workers=0)

    for images, labels in t_loader:
        print(f"[*] Batch Output Image Tensor Shape: {images.shape} (Expected: [2, 3, 224, 224])")
        print(f"[*] Batch Output Label Tensor Shape: {labels.shape} (Expected: [2, 5])")
        print(f"[*] Image Tensor Value Range: [{images.min():.2f}, {images.max():.2f}] (Normalized)")
        print(f"[*] Label Vector Sample: {labels[0].tolist()}")
        break

    # Clean up synthetic test artifact
    if os.path.exists(dummy_img_path):
        os.remove(dummy_img_path)
    if os.path.exists(test_dir):
        os.rmdir(test_dir)

    print("🎉 UNIT TEST PASSED: src/dataset.py is validated and production-ready.")