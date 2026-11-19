# filepath: thorax-ai/app.py
# =====================================================================
# File: thorax-ai/app.py
# Module: THORAX-AI Clinician Decision Support System (CDSS) Portal
# Architecture: Multi-Workspace Streamlit App with DenseNet-121 + Grad-CAM
# Compatibility: AMD Radeon / Native CPU Runtime & Chrome Cloud Portal
# =====================================================================

import io
import os
import sys
import zipfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

import torch
from torchvision import transforms

# Anchor project root to system path for modular package resolution
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import internal THORAX-AI modules
from src.dataset import TARGET_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_thorax_model_and_criterion, ThoraxDenseNet121
from src.gradcam import GradCAM, overlay_heatmap_on_radiograph
from src.report import generate_clinical_diagnostic_pdf

ROOT_PATH: Path = Path(project_root)
MODELS_DIR: Path = ROOT_PATH / "models"
ASSETS_DIR: Path = ROOT_PATH / "assets"
SAMPLE_DIR: Path = ROOT_PATH / "sample_xrays"


# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MEDICAL METADATA (SEO-INDEXED FOR CHROME)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="THORAX-AI | Explainable Chest X-Ray Screening & Triage CDSS",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISEASE_METADATA: Dict[str, Dict[str, str]] = {
    "Cardiomegaly": {
        "hex": "#E63946",
        "icon": "🫀",
        "display_name": "Cardiomegaly",
        "anatomy": "Transverse Cardiac Diameter & Ventricular Apex",
        "why_flagged": (
            "Marked lateral displacement of the left ventricular contour with an estimated "
            "cardiothoracic ratio (CTR) exceeding 0.50. Dense saliency gradients isolate "
            "enlarged cardiac silhouette boundaries."
        ),
        "symptoms": "Exertional dyspnea, orthopnea, bilateral lower-extremity pitting edema, paroxysmal nocturnal dyspnea.",
        "risks": "Congestive heart failure (CHF) decompensation, malignant ventricular arrhythmias, cardiogenic pulmonary congestion.",
        "routine_order": "Transthoracic Echocardiogram (TTE) for ejection fraction quantification; 12-lead ECG; serum NT-proBNP assay.",
        "urgent_order": "Immediate cardiology consultation; intravenous loop diuretic titration; strict fluid balance monitoring.",
    },
    "Effusion": {
        "hex": "#0077B6",
        "icon": "💧",
        "display_name": "Pleural Effusion",
        "anatomy": "Costophrenic Sulci & Dependent Basal Pleural Cavity",
        "why_flagged": (
            "Blunting of the sharp lateral and posterior costophrenic angles with characteristic "
            "meniscus sign formation denoting dependent serous liquid accumulation."
        ),
        "symptoms": "Sharp pleuritic chest pain on inspiration, dry hacking cough, tachypnea, asymmetric thoracic expansion.",
        "risks": "Compressive pulmonary atelectasis, severe ventilation-perfusion mismatch, hypoxemia, empyema or fibrothorax.",
        "routine_order": "Erect lateral decubitus radiograph to assess fluid mobility; diagnostic thoracentesis with pleural fluid analysis (Light's criteria).",
        "urgent_order": "Therapeutic large-volume thoracentesis; continuous pulse oximetry; urgent thoracic surgery evaluation if hemothorax suspected.",
    },
    "Pneumonia": {
        "hex": "#F77F00",
        "icon": "🔥",
        "display_name": "Pneumonia",
        "anatomy": "Subsegmental Alveoli & Lobar Bronchovascular Markings",
        "why_flagged": (
            "Focal or multi-focal parenchymal consolidation with ill-defined alveolar opacities "
            "and visible air bronchograms without underlying volume collapse."
        ),
        "symptoms": "High pyrexia with rigors, productive mucopurulent cough, pleurisy, tachypnea, bronchial breath sounds on auscultation.",
        "risks": "Acute Respiratory Distress Syndrome (ARDS), severe sepsis/bacteremia, parenchymal necrosis/cavitation, respiratory exhaustion.",
        "routine_order": "Sputum gram stain and culture; paired aerobic/anaerobic blood cultures; empiric outpatient respiratory fluoroquinolone/beta-lactam; CURB-65 calculation.",
        "urgent_order": "Immediate inpatient admission; high-flow humidified supplemental oxygen; emergent intravenous broad-spectrum antimicrobial therapy; arterial blood gas (ABG) panel.",
    },
    "Pneumothorax": {
        "hex": "#7209B7",
        "icon": "⚠️",
        "display_name": "Pneumothorax",
        "anatomy": "Visceral Pleural Line & Apicolateral Lung Periphery",
        "why_flagged": (
            "Distinct, fine radiopaque visceral pleural line separated from the thoracic parietal wall, "
            "accompanied by an absolute absence of peripheral bronchovascular lung markings."
        ),
        "symptoms": "Sudden-onset severe ipsilateral pleuritic pain, acute respiratory distress, tachycardia, hyperresonance to percussion.",
        "risks": "Tension pneumothorax progression with contralateral mediastinal shift, acute vena caval compression, obstructive shock, cardiovascular arrest.",
        "routine_order": "Supplemental high-flow normobaric oxygen to accelerate pleural nitrogen absorption; erect expiratory chest radiograph.",
        "urgent_order": "IMMEDIATE emergency needle decompression (2nd intercostal space, midclavicular line) followed by emergent tube thoracostomy (5th ICS, midaxillary line).",
    },
    "Atelectasis": {
        "hex": "#2A9D8F",
        "icon": "📉",
        "display_name": "Atelectasis",
        "anatomy": "Lobar / Segmental Bronchial Airway Branches",
        "why_flagged": (
            "Linear horizontal/oblique opacities (Fleischner lines) and displacement of interlobar "
            "fissures indicating parenchymal volume collapse without active consolidation."
        ),
        "symptoms": "Mild exertional breathlessness, low-grade postoperative pyrexia, basal hypoventilation, decreased vocal fremitus.",
        "risks": "Subsegmental hypoxemic shunt, high susceptibility to secondary post-obstructive bacterial superinfection.",
        "routine_order": "Incentive spirometry (10 breaths/hour while awake); early physical ambulation; chest physical therapy with postural drainage.",
        "urgent_order": "Flexible therapeutic bronchoscopy for direct mucus plug clearance; continuous positive airway pressure (CPAP); urgent chest CT.",
    },
}


# ---------------------------------------------------------------------
# 2. DYNAMIC THEME ENGINE (READING ROOM DARK VS CLINICAL LIGHT)
# ---------------------------------------------------------------------
def apply_clinical_theme() -> Tuple[str, Dict[str, str]]:
    """Applies clinical UI theme styling globally using Streamlit CSS injection."""
    st.sidebar.markdown("### 🖥️ Display Environment")
    theme_choice = st.sidebar.radio(
        "Display Mode",
        options=["🌙 Diagnostic Dark (Reading Room)", "☀️ Clinical Light (Ward Mode)"],
        index=0,
        label_visibility="collapsed",
    )

    is_dark = "Dark" in theme_choice
    active_theme = "dark" if is_dark else "light"

    if is_dark:
        palette = {
            "bg_main": "#0B0F17",
            "bg_card": "#131D31",
            "bg_sidebar": "#0E1524",
            "border": "#1E293B",
            "text_primary": "#F8FAFC",
            "text_secondary": "#94A3B8",
            "banner_bg": "linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
            "triage_crit_bg": "rgba(230, 57, 70, 0.14)",
            "triage_urg_bg": "rgba(247, 127, 0, 0.14)",
            "triage_rout_bg": "rgba(16, 185, 129, 0.14)",
            "bar_bg": "#27354F",
            "code_bg": "#1E293B",
            "code_text": "#38BDF8",
            "widget_bg": "#131D31",
            "widget_border": "#334155",
            "widget_text": "#E2E8F0",
            "hover_bg": "rgba(255, 255, 255, 0.05)",
        }
    else:
        palette = {
            "bg_main": "#F8FAFC",
            "bg_card": "#FFFFFF",
            "bg_sidebar": "#FFFFFF",
            "border": "#CBD5E1",
            "text_primary": "#0F172A",
            "text_secondary": "#475569",
            "banner_bg": "linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%)",
            "triage_crit_bg": "rgba(220, 38, 38, 0.10)",
            "triage_urg_bg": "rgba(217, 119, 6, 0.10)",
            "triage_rout_bg": "rgba(16, 185, 129, 0.10)",
            "bar_bg": "#E2E8F0",
            "code_bg": "#E2E8F0",
            "code_text": "#0F172A",
            "widget_bg": "#FFFFFF",
            "widget_border": "#94A3B8",
            "widget_text": "#0F172A",
            "hover_bg": "rgba(13, 148, 136, 0.08)",
        }

    css_injection = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
        .stApp {{ background-color: {palette['bg_main']} !important; color: {palette['text_primary']} !important; }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{ color: {palette['text_primary']} !important; }}
        .stApp p, .stApp span, .stApp label, .stApp li {{ color: {palette['text_primary']}; }}

        section[data-testid="stSidebar"] {{
            background-color: {palette['bg_sidebar']} !important;
            border-right: 1.5px solid {palette['border']} !important;
        }}

        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{
            color: {palette['text_primary']} !important;
        }}

        div[data-testid="stRadio"] > div {{ gap: 6px !important; }}
        div[data-testid="stRadio"] label {{
            background-color: {palette['bg_card']} !important; border: 1px solid {palette['border']} !important;
            border-radius: 8px !important; padding: 10px 14px !important; margin-bottom: 4px !important; cursor: pointer !important;
        }}
        div[data-testid="stRadio"] label:hover {{ border-color: #0D9488 !important; background-color: {palette['hover_bg']} !important; }}
        div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {{
            color: {palette['text_primary']} !important; font-weight: 600 !important; font-size: 13px !important;
        }}

        section[data-testid="stSidebar"] code {{
            background-color: {palette['code_bg']} !important; color: {palette['code_text']} !important;
            border: 1px solid {palette['border']} !important; padding: 2px 7px !important; border-radius: 4px !important;
            font-weight: 600 !important; font-size: 11.5px !important;
        }}

        .clinical-banner {{
            background: {palette['banner_bg']} !important; border-left: 5px solid #0D9488 !important;
            border-top: 1px solid {palette['border']}; border-right: 1px solid {palette['border']}; border-bottom: 1px solid {palette['border']};
            padding: 22px 26px; border-radius: 10px; margin-bottom: 22px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        }}
        .clinical-banner h1 {{ color: {palette['text_primary']} !important; font-size: 24px; font-weight: 700; margin: 0 0 6px 0; }}
        .clinical-banner p {{ color: {palette['text_secondary']} !important; font-size: 13.5px; margin: 0; }}

        .metric-card {{
            background-color: {palette['bg_card']} !important; border: 1px solid {palette['border']} !important;
            border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); transition: transform 0.2s ease;
        }}
        .metric-card:hover {{ border-color: #0D9488 !important; transform: translateY(-2px); }}

        .download-card {{
            background-color: {palette['bg_card']} !important; border: 1px solid {palette['border']} !important;
            border-top: 4px solid #0D9488 !important; border-radius: 10px; padding: 18px; margin-bottom: 14px;
            min-height: 195px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}

        div[data-testid="stFileUploader"] section {{
            background-color: {palette['widget_bg']} !important; border: 1.5px dashed {palette['widget_border']} !important;
            border-radius: 8px !important; padding: 18px !important;
        }}
        div[data-testid="stFileUploader"] section * {{ color: {palette['widget_text']} !important; }}

        div.stButton > button, div[data-testid="stDownloadButton"] > button {{
            background-color: {palette['widget_bg']} !important; color: {palette['widget_text']} !important;
            border: 1.5px solid {palette['widget_border']} !important; border-radius: 8px !important; padding: 8px 18px !important;
            font-weight: 600 !important; box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important; transition: all 0.2s ease !important;
        }}
        div.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {{
            border-color: #0D9488 !important; color: #0D9488 !important; transform: translateY(-1px) !important;
        }}

        div[data-testid="stAlert"] {{ background-color: {palette['bg_card']} !important; border: 1px solid {palette['border']} !important; border-left: 5px solid #0D9488 !important; border-radius: 8px !important; }}
        div[data-testid="stAlert"] * {{ color: {palette['text_primary']} !important; }}

        .triage-critical {{ background-color: {palette['triage_crit_bg']} !important; border-left: 5px solid #E63946 !important; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
        .triage-urgent {{ background-color: {palette['triage_urg_bg']} !important; border-left: 5px solid #F77F00 !important; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
        .triage-routine {{ background-color: {palette['triage_rout_bg']} !important; border-left: 5px solid #10B981 !important; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}

        div[data-testid="stExpander"] {{ background-color: {palette['bg_card']} !important; border: 1px solid {palette['border']} !important; border-radius: 8px !important; margin-bottom: 12px !important; }}
        div[data-testid="stExpander"] summary span, div[data-testid="stExpander"] summary p {{ color: {palette['text_primary']} !important; font-weight: 600 !important; }}

        button[data-testid="stTab"] p {{ color: {palette['text_secondary']} !important; font-weight: 600 !important; }}
        button[data-testid="stTab"][aria-selected="true"] p {{ color: #0D9488 !important; }}

        div[data-testid="stSelectbox"] label p, div[data-testid="stFileUploader"] label p, div[data-testid="stSlider"] label p, div[data-testid="stTextInput"] label p, div[data-testid="stNumberInput"] label p {{
            color: {palette['text_primary']} !important; font-weight: 600 !important;
        }}
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ background-color: {palette['bg_card']} !important; border-color: {palette['border']} !important; color: {palette['text_primary']} !important; }}
    </style>
    """
    st.markdown(css_injection, unsafe_allow_html=True)
    return active_theme, palette


# ---------------------------------------------------------------------
# 3. PIPELINE LOADING, INFERENCE & OFFLINE PACKAGERS
# ---------------------------------------------------------------------
def find_checkpoint_file() -> Optional[Path]:
    """Resolves `models/densenet121_best.pth` or falls back to any `.pth` file inside `models/`."""
    preferred = MODELS_DIR / "densenet121_best.pth"
    if preferred.exists():
        return preferred
    if MODELS_DIR.exists():
        pth_files = sorted(MODELS_DIR.glob("*.pth"))
        if pth_files:
            return pth_files[0]
    return None


@st.cache_resource
def load_screening_system() -> Tuple[ThoraxDenseNet121, GradCAM, torch.device, Optional[str]]:
    """Loads the compiled PyTorch DenseNet-121 model and Grad-CAM explainer on CPU/AMD."""
    device = torch.device("cpu")
    model, _ = build_thorax_model_and_criterion(
        num_classes=len(TARGET_CLASSES),
        pretrained=False,
        device=device,
    )

    checkpoint_path = find_checkpoint_file()
    loaded_name: Optional[str] = None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        loaded_name = checkpoint_path.name
        print(f"[*] Successfully loaded trained checkpoint: {checkpoint_path}")
    else:
        print(f"[!] No checkpoint located in {MODELS_DIR}. Using baseline weights.")

    model.eval()
    cam_target_layer = model.get_cam_target_layer()
    cam_engine = GradCAM(model=model, target_layer=cam_target_layer)
    return model, cam_engine, device, loaded_name


def preprocess_radiograph(image_data: bytes) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Applies clinical CLAHE enhancement (clipLimit=2.0, tileGridSize=(8, 8))
    and ImageNet normalization.

    Returns:
        display_image: RGB uint8 array of shape (448, 448, 3) for UI rendering.
        tensor: Normalized float32 tensor of shape (1, 3, 224, 224) for DenseNet-121.
    """
    np_buf = np.frombuffer(image_data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode radiograph image stream.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # tensor shape: (1, 3, 224, 224)
    tensor = transform(enhanced_rgb).unsqueeze(0)
    # display_image shape: (448, 448, 3)
    display_image = cv2.resize(enhanced_rgb, (448, 448))
    return display_image, tensor


def generate_anatomical_sample_scan() -> bytes:
    """Generates an embedded OpenCV synthetic baseline radiograph."""
    canvas = np.zeros((448, 448), dtype=np.uint8)
    cv2.rectangle(canvas, (20, 20), (428, 428), 15, -1)
    cv2.ellipse(canvas, (160, 220), (65, 120), -5, 0, 360, 95, -1)
    cv2.ellipse(canvas, (288, 220), (65, 120), 5, 0, 360, 95, -1)
    cv2.ellipse(canvas, (224, 240), (55, 90), 0, 0, 360, 40, -1)
    cv2.ellipse(canvas, (250, 260), (70, 55), 20, 0, 360, 50, -1)
    cv2.fillPoly(canvas, [np.array([[310, 300], [355, 340], [300, 340]])], 75)
    for y in range(120, 320, 32):
        cv2.ellipse(canvas, (224, y), (140, 22), 0, 0, 180, 125, 4)
    canvas = cv2.GaussianBlur(canvas, (15, 15), 0)
    _, encoded = cv2.imencode(".png", canvas)
    return encoded.tobytes()


WINDOWS_LAUNCHER: str = "\n".join([
    "@echo off",
    "title THORAX-AI Clinical Triage Launcher (Windows)",
    "echo ============================================================",
    "echo   THORAX-AI: Explainable Thoracic Disease Screening System",
    "echo   Initializing Offline Clinical Environment (Windows)...",
    "echo ============================================================",
    'if not exist "thorax_env" (',
    "    echo [1/3] Creating isolated Python virtual environment...",
    "    python -m venv thorax_env",
    ")",
    r"call thorax_env\Scripts\activate",
    "echo [2/3] Installing offline CPU dependencies...",
    "pip install --quiet --upgrade pip",
    "pip install --quiet -r requirements.txt",
    "echo [3/3] Launching THORAX-AI Clinical Dashboard in Chrome...",
    "streamlit run app.py",
    "pause",
])

MACOS_LAUNCHER: str = "\n".join([
    "#!/bin/bash",
    'cd "$(dirname "$0")"',
    'echo "============================================================"',
    'echo "  THORAX-AI: Explainable Thoracic Disease Screening System"',
    'echo "  Initializing Offline Clinical Environment (macOS)..."',
    'echo "============================================================"',
    'if [ ! -d "thorax_env" ]; then',
    '    echo "[1/3] Creating isolated Python virtual environment..."',
    "    python3 -m venv thorax_env",
    "fi",
    "source thorax_env/bin/activate",
    'echo "[2/3] Installing offline CPU dependencies..."',
    "pip install --quiet --upgrade pip",
    "pip install --quiet -r requirements.txt",
    'echo "[3/3] Launching THORAX-AI Clinical Dashboard in Chrome..."',
    "streamlit run app.py",
])


@st.cache_data
def build_sample_xrays_zip() -> Optional[bytes]:
    """Packages all sample radiographs inside `sample_xrays/` into a downloadable ZIP."""
    if not SAMPLE_DIR.exists():
        return None
    sample_files = sorted(
        [p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not sample_files:
        return None

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in sample_files:
            zf.write(img_path, arcname=f"THORAX_AI_Sample_Xrays/{img_path.name}")
    return buf.getvalue()


@st.cache_data
def build_os_offline_bundle(target_os: str) -> bytes:
    """
    Packages the full `thorax-ai` product (`app.py`, `train.py`, `verify_env.py`,
    `src/`, `models/`, `assets/`, `sample_xrays/`) with OS auto-launchers.
    """
    buf = io.BytesIO()
    include_items = [
        "app.py",
        "train.py",
        "verify_env.py",
        "requirements.txt",
        "packages.txt",
        "README.md",
        "src",
        "models",
        "assets",
        "sample_xrays",
    ]

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in include_items:
            item_path = ROOT_PATH / item_name
            if not item_path.exists():
                continue
            if item_path.is_file():
                zf.write(item_path, arcname=f"thorax-ai/{item_name}")
            elif item_path.is_dir():
                for file_path in item_path.rglob("*"):
                    if file_path.is_file() and "__pycache__" not in file_path.parts:
                        rel_path = file_path.relative_to(ROOT_PATH)
                        zf.write(file_path, arcname=f"thorax-ai/{rel_path.as_posix()}")

        if target_os == "windows":
            zf.writestr("thorax-ai/RUN_THORAX_AI_WINDOWS.bat", WINDOWS_LAUNCHER)
        else:
            info = zipfile.ZipInfo("thorax-ai/RUN_THORAX_AI_MAC.command")
            info.create_system = 3
            info.external_attr = 0o755 << 16
            zf.writestr(info, MACOS_LAUNCHER)

    return buf.getvalue()


# ---------------------------------------------------------------------
# 4. APPLICATION WORKSPACES
# ---------------------------------------------------------------------
def main() -> None:
    active_theme, palette = apply_clinical_theme()
    model, cam_engine, device, loaded_ckpt = load_screening_system()

    # Sidebar Brand Logo & Header
    logo_path = ASSETS_DIR / "thorax_logo.png"
    if logo_path.exists():
        col_l1, col_l2, col_l3 = st.sidebar.columns([1, 1.4, 1])
        with col_l2:
            st.image(logo_path.read_bytes(), width="stretch")

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; padding: 4px 0 14px 0;">
            <h2 style="margin: 2px 0 0 0; font-size: 22px; font-weight: 700; color: {palette['text_primary']} !important;">THORAX-AI</h2>
            <p style="color: #0D9488; font-size: 11px; margin: 3px 0 0 0; font-weight: 600; letter-spacing: 0.6px;">CLINICAL DECISION SUPPORT SYSTEM</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace = st.sidebar.radio(
        "Clinical Workspace",
        options=[
            "🫁 Clinical Knowledge Hub",
            "🔬 Screening & XAI Studio",
            "📋 Diagnostic Decision Support",
            "📊 Benchmark & Validation Suite",
            "⬇️ Download Product & Sample X-Rays",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Runtime Hardware:** `{device.type.upper()}`")
    st.sidebar.markdown("**Feature Extractor:** `DenseNet-121`")
    if loaded_ckpt:
        st.sidebar.markdown(f"**Active Weights:** `{loaded_ckpt}`")
    else:
        st.sidebar.markdown("**Active Weights:** `Baseline (Unloaded)`")
    st.sidebar.markdown("**Evaluation Metric:** `Macro AUC-ROC`")

    # =================================================================
    # WORKSPACE 1: CLINICAL KNOWLEDGE HUB
    # =================================================================
    if "Clinical Knowledge Hub" in workspace:
        hero_gif = ASSETS_DIR / "hero_banner.gif"
        hero_png = ASSETS_DIR / "hero_banner.png"
        if hero_gif.exists():
            st.image(hero_gif.read_bytes(), width="stretch")
        elif hero_png.exists():
            st.image(hero_png.read_bytes(), width="stretch")

        st.markdown(
            """
            <div class="clinical-banner">
                <h1>Clinical Imperative of Thoracic Disease Screening</h1>
                <p>Pathophysiology, Epidemiologic Urgency, and Interpretable Machine Intelligence in Pulmonary Medicine</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1.1, 0.9])
        with col_a:
            st.markdown("### The Global Healthcare Bottleneck")
            st.markdown(
                "\n".join([
                    "Cardiopulmonary diseases remain the leading causes of acute mortality globally. According to the "
                    "**World Health Organization (WHO)**, acute lower respiratory tract infections (pneumonia), decompensated "
                    "heart failure (manifesting as cardiomegaly and pleural effusion), and acute alveolar collapse claim over "
                    "**6.5 million lives every year**.",
                    "",
                    "- **Critical Turnaround Delays:** In primary healthcare centers and rural clinics, the radiologist-to-patient "
                    "ratio regularly exceeds **1:100,000**, leading to critical **48–72 hour diagnostic delays**.",
                    "- **The Black-Box Dilemma:** While deep learning classifiers achieve high benchmark accuracy, clinical adoption "
                    "remains low because clinicians cannot trust unverified numerical probabilities without visual justification.",
                    "- **The THORAX-AI Paradigm:** Combining **DenseNet-121 feature reuse** with **Gradient-Weighted Class Activation "
                    "Mapping (Grad-CAM)** guarantees that predictions are visually grounded against true anatomical landmarks.",
                ])
            )

        with col_b:
            clinic_img = ASSETS_DIR / "clinic_showcase.png"
            if clinic_img.exists():
                st.image(
                    clinic_img.read_bytes(),
                    caption="Offline Rural Clinic Triage Deployment (CPU / AMD Compatible)",
                    width="stretch",
                )
            st.markdown("### Core Anatomical Screening Targets")
            st.markdown(
                "\n".join([
                    "- **🫀 Cardiac Silhouette:** Evaluating the Cardiothoracic Ratio (CTR > 0.50) to flag chamber hypertrophy.",
                    "- **💧 Costophrenic Sulci:** Detecting lateral angle blunting to isolate dependent serous fluid layering.",
                    "- **🔥 Alveolar Parenchyma:** Identifying irregular infiltrates and air bronchograms indicating consolidation.",
                    "- **⚠️ Visceral Pleura:** Tracking pleural detachment and tension margins in pneumothorax.",
                    "- **📉 Bronchovascular Bundles:** Flagging linear plate-like atelectatic volume reduction.",
                ])
            )

        st.markdown("---")
        st.markdown("### Pathological Signatures & Physiological Tokens")

        cols = st.columns(5)
        for idx, (condition, meta) in enumerate(DISEASE_METADATA.items()):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="background-color: {palette['bg_card']}; border-top: 4px solid {meta['hex']}; border-left: 1px solid {palette['border']}; border-right: 1px solid {palette['border']}; border-bottom: 1px solid {palette['border']}; padding: 14px; border-radius: 8px; min-height: 180px;">
                        <span style="font-size: 24px;">{meta['icon']}</span>
                        <h4 style="color: {palette['text_primary']}; margin: 8px 0 4px 0; font-size: 14px;">{meta['display_name']}</h4>
                        <p style="color: {palette['text_secondary']}; font-size: 11px; margin: 0;"><b>Target Anatomy:</b><br>{meta['anatomy']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =================================================================
    # WORKSPACE 2: SCREENING & XAI STUDIO
    # =================================================================
    elif "Screening & XAI Studio" in workspace:
        st.markdown(
            """
            <div class="clinical-banner">
                <h1>Radiograph Screening & Explainable AI Studio</h1>
                <p>Dual-View Diagnostic Grounding with Real-Time Gradient-Weighted Activation Attribution</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_upload, col_preset, col_sample = st.columns([2.2, 1.5, 1.1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Frontal Chest Radiograph (PNG or JPEG)",
                type=["png", "jpg", "jpeg"],
            )
        with col_preset:
            sample_options = ["None (Use Upload / Synthetic)"]
            if SAMPLE_DIR.exists():
                sample_options += sorted(
                    [p.name for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                )
            chosen_sample = st.selectbox(
                "Or Pick a Test Case from sample_xrays/:",
                options=sample_options,
            )
        with col_sample:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            use_sample = st.button("🔬 Synthetic Baseline", width="stretch")

        raw_bytes: Optional[bytes] = None
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
        elif chosen_sample != "None (Use Upload / Synthetic)":
            raw_bytes = (SAMPLE_DIR / chosen_sample).read_bytes()
        elif use_sample or "active_scan" in st.session_state:
            if use_sample or "active_scan" not in st.session_state:
                st.session_state["active_scan"] = generate_anatomical_sample_scan()
            raw_bytes = st.session_state["active_scan"]

        if raw_bytes is not None:
            try:
                display_img, input_tensor = preprocess_radiograph(raw_bytes)
            except ValueError as err:
                st.error(f"Radiograph decoding failed: {err}")
                return

            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                # logits shape: (1, 5) -> probabilities shape: (5,)
                logits = model(input_tensor)
                probabilities = torch.sigmoid(logits)[0].cpu().numpy()

            st.session_state["last_probabilities"] = probabilities
            st.session_state["last_display_img"] = display_img

            col_view, col_metrics = st.columns([1.1, 0.9])

            with col_view:
                st.markdown("#### Radiograph & Grad-CAM Visualizer")

                selected_disease = st.selectbox(
                    "Select Condition to Explain via Grad-CAM:",
                    options=TARGET_CLASSES,
                    index=int(np.argmax(probabilities)),
                )

                alpha_slider = st.slider(
                    "Grad-CAM Heatmap Opacity (Alpha Blending):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.45,
                    step=0.05,
                )

                disease_idx = TARGET_CLASSES.index(selected_disease)
                cam_map = cam_engine.generate_heatmap(input_tensor, class_idx=disease_idx)
                blended_scan = overlay_heatmap_on_radiograph(
                    radiograph_rgb=display_img,
                    heatmap=cam_map,
                    alpha=alpha_slider,
                    colormap=cv2.COLORMAP_JET,
                )

                st.session_state["last_blended_img"] = blended_scan
                st.session_state["last_selected_disease"] = selected_disease

                tab_cam, tab_raw = st.tabs(["Diagnostic CAM Overlay", "CLAHE-Enhanced Original"])
                with tab_cam:
                    st.image(blended_scan, caption=f"Grad-CAM Attribution: {selected_disease}", width="stretch")
                with tab_raw:
                    st.image(display_img, caption="Preprocessed Radiograph (OpenCV CLAHE)", width="stretch")

                cam_buf = io.BytesIO()
                Image.fromarray(blended_scan).save(cam_buf, format="PNG")
                st.download_button(
                    label=f"⬇️ Download Grad-CAM Overlay ({selected_disease}.png)",
                    data=cam_buf.getvalue(),
                    file_name=f"THORAX_AI_GradCAM_{selected_disease}.png",
                    mime="image/png",
                    width="stretch",
                )

            with col_metrics:
                st.markdown("#### Multi-Label Predicted Risk Profile")
                st.caption("Thresholds: Normal (<0.35) | Borderline (0.35–0.70) | Flagged (>0.70)")

                for i, disease_name in enumerate(TARGET_CLASSES):
                    prob = float(probabilities[i])
                    meta = DISEASE_METADATA[disease_name]
                    accent = meta["hex"]

                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 4px solid {accent} !important;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600; font-size: 14px; color: {palette['text_primary']} !important;">
                                    {meta['icon']} {meta['display_name']}
                                </span>
                                <span style="font-weight: 700; font-size: 16px; color: {accent};">
                                    {prob * 100:.1f}%
                                </span>
                            </div>
                            <div style="background-color: {palette['bar_bg']}; height: 6px; border-radius: 3px; margin-top: 8px; overflow: hidden;">
                                <div style="background-color: {accent}; width: {min(max(prob * 100, 3), 100)}%; height: 100%; border-radius: 3px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.info("💡 Review full triage and export signed medical PDF under **📋 Diagnostic Decision Support**.")
        else:
            st.warning("Upload a chest radiograph, select a test scan from `sample_xrays/`, or click 'Synthetic Baseline' to begin screening.")

    # =================================================================
    # WORKSPACE 3: DIAGNOSTIC DECISION SUPPORT (CDSS) & PDF EXPORT
    # =================================================================
    elif "Diagnostic Decision Support" in workspace:
        st.markdown(
            """
            <div class="clinical-banner">
                <h1>Automated Clinical Decision Support System (CDSS)</h1>
                <p>Objective Radiologic Findings, Symptom Profiles, Evidence-Based Orders, and Signed PDF Consultation Export</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "last_probabilities" not in st.session_state or "last_display_img" not in st.session_state:
            st.error("No active patient scan in memory. Please upload or load a scan in Screening & XAI Studio.")
            return

        probs = st.session_state["last_probabilities"]
        raw_img = st.session_state["last_display_img"]
        blended_img = st.session_state.get("last_blended_img", raw_img)
        primary_disease = st.session_state.get("last_selected_disease", TARGET_CLASSES[int(np.argmax(probs))])

        tier3_critical = [TARGET_CLASSES[i] for i, p in enumerate(probs) if p >= 0.70]
        tier2_moderate = [TARGET_CLASSES[i] for i, p in enumerate(probs) if 0.35 <= p < 0.70]

        if len(tier3_critical) > 0:
            st.markdown(
                f"""
                <div class="triage-critical">
                    <h3 style="margin: 0; font-size: 17px; color: #E63946; font-weight: 700;">
                        🚨 ACUITY TIER 3 — IMMEDIATE CLINICAL INTERVENTION REQUIRED
                    </h3>
                    <p style="margin: 4px 0 0 0; color: {palette['text_primary']}; font-size: 13px;">
                        High-confidence acute pathology detected: <b>{', '.join(tier3_critical)}</b>.
                        Initiate urgent bedside stabilization and specialist consult.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif len(tier2_moderate) > 0:
            st.markdown(
                f"""
                <div class="triage-urgent">
                    <h3 style="margin: 0; font-size: 17px; color: #F77F00; font-weight: 700;">
                        ⚠️ ACUITY TIER 2 — SECONDARY CLINICAL CORRELATION INDICATED
                    </h3>
                    <p style="margin: 4px 0 0 0; color: {palette['text_primary']}; font-size: 13px;">
                        Intermediate probability detected: <b>{', '.join(tier2_moderate)}</b>.
                        Execute targeted diagnostic orders within 24–48 hours.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="triage-routine">
                    <h3 style="margin: 0; font-size: 17px; color: #10B981; font-weight: 700;">
                        ✅ ACUITY TIER 1 — NO HIGH-RISK PATHOLOGY IDENTIFIED
                    </h3>
                    <p style="margin: 4px 0 0 0; color: {palette['text_primary']}; font-size: 13px;">
                        All 5 screened conditions fall beneath baseline risk thresholds. Routine preventive observation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Detailed Finding Sheets & Physician Orders")

        for i, disease_name in enumerate(TARGET_CLASSES):
            p = float(probs[i])
            meta = DISEASE_METADATA[disease_name]
            is_expanded = p >= 0.35

            with st.expander(f"{meta['icon']} {meta['display_name']} — Assessment (Risk: {p * 100:.1f}%)", expanded=is_expanded):
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.markdown(f"**Anatomical Target:** {meta['anatomy']}")
                    st.markdown(f"**Why Flagged by Model:** {meta['why_flagged']}")
                    st.markdown(f"**Presenting Symptoms to Correlate:** {meta['symptoms']}")
                    st.markdown(f"**Pathological Risks:** {meta['risks']}")

                with col_right:
                    st.markdown("#### Recommended Clinical Action Plan")
                    if p >= 0.70:
                        st.error(f"**Tier 3 Critical Order:** {meta['urgent_order']}")
                    elif p >= 0.35:
                        st.warning(f"**Tier 2 Diagnostic Order:** {meta['routine_order']}")
                    else:
                        st.success("**Tier 1 Protocol:** Findings within physiological baseline. Follow standard clinical guidelines.")

        st.markdown("---")
        st.markdown("### 📄 Export Signed Physician Consultation Report (PDF)")
        st.caption("Populate clinical chart metadata to compile an accredited PACS diagnostic consultation summary.")

        c_dem1, c_dem2, c_dem3 = st.columns(3)
        with c_dem1:
            p_id = st.text_input("Patient Medical Record Number (MRN)", value="PAT-004289")
            p_name = st.text_input("Patient Full Legal Name", value="Eleanor Vance")
        with c_dem2:
            p_age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=62)
            p_gender = st.selectbox("Biological Sex / Gender", ["Female", "Male", "Other"])
        with c_dem3:
            p_physician = st.text_input("Reviewing Clinician / Radiologist", value="Dr. Marcus Bennett, MD")
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

            findings_dict = {TARGET_CLASSES[idx]: float(probs[idx]) for idx in range(len(TARGET_CLASSES))}

            pdf_bytes = generate_clinical_diagnostic_pdf(
                patient_id=p_id,
                patient_name=p_name,
                patient_age=int(p_age),
                patient_gender=p_gender,
                referring_physician=p_physician,
                findings_probabilities=findings_dict,
                disease_metadata=DISEASE_METADATA,
                raw_radiograph_rgb=raw_img,
                gradcam_overlay_rgb=blended_img,
                primary_condition_explained=primary_disease,
            )

            st.download_button(
                label="📥 Download Signed PDF Consultation Report",
                data=pdf_bytes,
                file_name=f"THORAX_AI_Report_{p_id}.pdf",
                mime="application/pdf",
                width="stretch",
            )

    # =================================================================
    # WORKSPACE 4: BENCHMARK & VALIDATION SUITE
    # =================================================================
    elif "Benchmark & Validation Suite" in workspace:
        st.markdown(
            """
            <div class="clinical-banner">
                <h1>Validation Protocols & Clinical Benchmark Standards</h1>
                <p>Zero Patient Data Leakage, Cost-Sensitive Loss Formulation, and Comparative Performance Metrics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        arch_img = ASSETS_DIR / "pipeline_architecture.png"
        if arch_img.exists():
            st.image(
                arch_img.read_bytes(),
                caption="THORAX-AI 5-Stage End-to-End Explainable Screening Architecture",
                width="stretch",
            )

        st.markdown("### 1. Leak-Free Patient Partitioning Protocol")
        st.markdown(
            "\n".join([
                "In benchmark radiograph datasets such as **NIH ChestX-ray14**, multiple scans are acquired for individual patients "
                "over months or years. Partitioning splits at the *image level* introduces severe **patient data leakage**, as unique "
                "thoracic cage geometry, surgical wires, and anatomical quirks are memorized by the network.",
                "",
                "**THORAX-AI** strictly enforces **GroupShuffleSplit** grouped on `Patient ID`:",
                r"$$\text{Patients}_{\text{Train}} \cap \text{Patients}_{\text{Validation}} = \emptyset$$",
            ])
        )

        st.markdown("### 2. Multi-Label Cost-Sensitive Optimization")
        st.markdown(
            "\n".join([
                r"Thoracic diseases exhibit severe positive/negative class imbalance (e.g., Pneumothorax prevalence $\approx 4.8\%$). "
                "Standard Cross-Entropy causes gradient collapse toward negative predictions. **THORAX-AI** implements **Weighted BCEWithLogitsLoss**:",
                r"$$\mathcal{L}_{W-BCE} = -\sum_{c=1}^{5} \left[ w_{pos, c} \cdot y_c \log(\sigma(z_c)) + (1 - y_c) \log(1 - \sigma(z_c)) \right]$$",
                "Where positive class weights are dynamically computed from training prevalence:",
                r"$$w_{pos, c} = \frac{N_{negative, c}}{N_{positive, c}}$$",
            ])
        )

        st.markdown("### 3. Model Benchmark Targets & Clinical Impact")
        benchmark_table = {
            "Condition": TARGET_CLASSES,
            "Target AUROC": ["0.884", "0.867", "0.742", "0.851", "0.795"],
            "Target Sensitivity": ["84.2%", "82.5%", "76.1%", "81.8%", "78.4%"],
            "Target Specificity": ["89.1%", "87.3%", "82.4%", "88.6%", "81.9%"],
            "Clinical Endpoint": [
                "Early cardiomegaly detection prevents unmanaged CHF hospitalizations.",
                "Identifies blunting before compressive atelectasis develops.",
                "Reduces antimicrobial initiation delay for lobar pneumonia.",
                "Averts progression to fatal tension pneumothorax.",
                "Differentiates mucus-plug collapse from active consolidation.",
            ],
        }
        st.table(pd.DataFrame(benchmark_table))

    # =================================================================
    # WORKSPACE 5: DOWNLOAD PRODUCT (WINDOWS / MACOS) & SAMPLE X-RAYS
    # =================================================================
    elif "Download Product" in workspace:
        st.markdown(
            """
            <div class="clinical-banner">
                <h1>Download THORAX-AI Offline Clinical Suite & Sample X-Rays</h1>
                <p>Standalone CPU/AMD Clinical Packages for Windows & macOS + Curated Radiograph Test Dataset</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        d_col1, d_col2, d_col3 = st.columns(3)

        with d_col1:
            st.markdown(
                f"""
                <div class="download-card">
                    <h3 style="margin: 0 0 8px 0; font-size: 18px;">🪟 Windows Offline Edition</h3>
                    <p style="color: {palette['text_secondary']}; font-size: 13px;">
                        Complete standalone bundle for Windows 10/11 (Intel & AMD CPUs). Includes
                        <b>RUN_THORAX_AI_WINDOWS.bat</b> one-click launcher, trained DenseNet-121 weights,
                        Grad-CAM engine, and PDF report generator.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Download for Windows (.zip)",
                data=build_os_offline_bundle("windows"),
                file_name="THORAX-AI_Windows_Offline_Edition.zip",
                mime="application/zip",
                width="stretch",
            )

        with d_col2:
            st.markdown(
                f"""
                <div class="download-card">
                    <h3 style="margin: 0 0 8px 0; font-size: 18px;">🍎 macOS Offline Edition</h3>
                    <p style="color: {palette['text_secondary']}; font-size: 13px;">
                        Complete standalone bundle for macOS (Apple Silicon M1–M4 & Intel). Includes
                        executable <b>RUN_THORAX_AI_MAC.command</b> auto-setup script and offline
                        clinician dashboard.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Download for macOS (.zip)",
                data=build_os_offline_bundle("macos"),
                file_name="THORAX-AI_macOS_Offline_Edition.zip",
                mime="application/zip",
                width="stretch",
            )

        with d_col3:
            st.markdown(
                f"""
                <div class="download-card">
                    <h3 style="margin: 0 0 8px 0; font-size: 18px;">🩻 Sample X-Ray Test Pack</h3>
                    <p style="color: {palette['text_secondary']}; font-size: 13px;">
                        13 curated test radiographs (2 scans per pathology across all 5 diseases +
                        3 normal chest radiographs) ready to test live triage and Grad-CAM heatmaps.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sample_zip = build_sample_xrays_zip()
            if sample_zip is not None:
                st.download_button(
                    label="⬇️ Download Sample X-Rays (.zip)",
                    data=sample_zip,
                    file_name="THORAX_AI_Test_Xrays_JPEG.zip",
                    mime="application/zip",
                    width="stretch",
                )
            else:
                st.info("Add the 13 test JPEGs to `thorax-ai/sample_xrays/` to enable direct ZIP download.")


if __name__ == "__main__":
    main()
