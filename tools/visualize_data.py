"""
Streamlit viewer for the bird-counting dataset.

Side-by-side UI:
- Left: the current image, with a Ground-Truth tab (annotation dots) and a
  Prediction tab (model density-map heatmap).
- Right: prev/next navigation, jump-to-index, model path input, and a
  "Run Prediction" button that compares predicted vs. ground-truth count.

Annotations are looked up in this order:
    1. <data_root>/annotations/<split>.json   (preferred; produced by
       tools/yolo_to_coco_converter.py — pixel-coord points keyed by stem).
    2. <data_root>/annotations/<split>/<stem>.txt   (legacy per-image:
       2 cols = pixel (x, y); 3 cols = YOLO (class, x_norm, y_norm)).

Run with:
    streamlit run tools/visualize_data.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# Make `models` importable when launched via `streamlit run tools/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models import get_model  # noqa: E402


warnings.filterwarnings("ignore")


# --- Configuration ---------------------------------------------------------
DEFAULT_DATA_ROOT = "../data"
DEFAULT_MODEL_PATH = "../ckpts/shufflenet_model_best.pth"
MODEL_TYPE = "shufflenet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training-time normalization (ImageNet stats).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CNN backbone requires inputs aligned to a multiple of this.
INPUT_ALIGNMENT = 32

# Heatmap rendering.
HEATMAP_THRESHOLD = 0.05  # mask out near-zero density to avoid blue wash
HEATMAP_ALPHA_BG = 0.6
HEATMAP_ALPHA_FG = 0.4
ERROR_TOLERANCE = 5.0  # |pred - gt| above this is shown red

IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png")

NORMALIZE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


# --- Data loading ----------------------------------------------------------
def list_images(root: Path, split: str) -> list[Path]:
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        return []
    files: list[Path] = []
    for pattern in IMAGE_GLOBS:
        files.extend(img_dir.glob(pattern))
    return sorted(files)


@st.cache_data(show_spinner=False)
def load_split_json(root: Path, split: str) -> dict[str, np.ndarray] | None:
    """Return {stem: (N, 2) pixel-coord array} for the split's JSON, or None if absent."""
    json_path = root / "annotations" / f"{split}.json"
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Failed to parse {json_path}: {e}")
        return None
    by_stem: dict[str, np.ndarray] = {}
    for ann in data.get("annotations", []):
        stem = str(ann.get("image_id", ""))
        pts = ann.get("points") or []
        by_stem[stem] = np.asarray(pts, dtype=float).reshape(-1, 2)
    return by_stem


def load_points_px(img_path: Path, root: Path, split: str, w: int, h: int) -> np.ndarray:
    """Pixel-coord points for `img_path`. Prefers the JSON split, falls back to legacy .txt."""
    by_stem = load_split_json(root, split)
    if by_stem is not None:
        return by_stem.get(img_path.stem, np.empty((0, 2)))
    return np.empty((0, 2))


# --- Inference -------------------------------------------------------------
@st.cache_resource
def load_model_cached(model_path: str):
    model = get_model(MODEL_TYPE, model_path, device=DEVICE, fuse=True)
    model.eval()
    return model


def run_inference(model, img_rgb: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (predicted_count, RGB image with heatmap overlay)."""
    orig_h, orig_w = img_rgb.shape[:2]
    img_tensor = NORMALIZE(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

    h, w = img_tensor.shape[2:]
    ph = (-h) % INPUT_ALIGNMENT
    pw = (-w) % INPUT_ALIGNMENT
    if ph or pw:
        img_tensor = F.pad(img_tensor, (0, pw, 0, ph))

    with torch.no_grad():
        mu, _ = model(img_tensor)

    pred_count = float(torch.sum(mu).item())
    density = mu.squeeze().cpu().numpy()
    overlay = blend_heatmap(img_rgb, density, orig_w, orig_h)
    return pred_count, overlay


def blend_heatmap(img_rgb: np.ndarray, density: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize density to image size, color-map JET, and alpha-blend over the image."""
    d_min, d_max = float(density.min()), float(density.max())
    if d_max > d_min:
        norm = (density - d_min) / (d_max - d_min + 1e-6)
    else:
        norm = np.zeros_like(density)
    norm = cv2.resize(norm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    colormap_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)

    overlay = img_rgb.copy()
    mask = norm > HEATMAP_THRESHOLD
    if mask.any():
        blended = cv2.addWeighted(img_rgb, HEATMAP_ALPHA_BG, colormap_rgb, HEATMAP_ALPHA_FG, 0)
        overlay[mask] = blended[mask]
    return overlay


# --- Drawing ---------------------------------------------------------------
def draw_ground_truth(img_rgb: np.ndarray, points_px: np.ndarray, pt_size: int) -> np.ndarray:
    out = img_rgb.copy()
    for cx, cy in points_px:
        cv2.circle(out, (int(cx), int(cy)), pt_size, (255, 0, 0), -1)
        cv2.circle(out, (int(cx), int(cy)), pt_size + 2, (255, 255, 255), 1)
    return out


# --- Streamlit state -------------------------------------------------------
def init_session():
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("pred_result", None)


def go_to(new_idx: int, total: int):
    st.session_state.idx = new_idx % total
    st.session_state.pred_result = None  # invalidate stale prediction


# --- UI --------------------------------------------------------------------
def render_sidebar() -> tuple[Path, str, int]:
    st.sidebar.title("🛠️ Settings")
    data_root = Path(st.sidebar.text_input("Dataset Root", value=DEFAULT_DATA_ROOT))

    images_dir = data_root / "images"
    if not images_dir.is_dir():
        st.error(f"Path not found: {images_dir}")
        st.stop()

    splits = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
    if not splits:
        st.error(f"No splits under {images_dir}")
        st.stop()

    split = st.sidebar.selectbox("Data Split", splits)
    pt_size = st.sidebar.slider("GT Point Size", 1, 10, 4)
    return data_root, split, pt_size


def render_image_panel(img_rgb: np.ndarray, points_px: np.ndarray, pt_size: int, name: str, idx: int, total: int):
    st.subheader(f"🖼️ {name} ({idx + 1}/{total})")
    gt_img = draw_ground_truth(img_rgb, points_px, pt_size)

    tab_gt, tab_pred = st.tabs(["📌 Ground Truth", "🔥 Model Prediction"])
    with tab_gt:
        st.image(gt_img, width="stretch", caption=f"GT Count: {len(points_px)}")
    with tab_pred:
        if st.session_state.pred_result is not None:
            p_count, p_overlay = st.session_state.pred_result
            st.image(p_overlay, width="stretch", caption=f"Predicted Count: {p_count:.2f}")
        else:
            st.info("Run model prediction from the sidebar to view results.")


def render_controls(img_rgb: np.ndarray, total: int, gt_count: int):
    st.write("### 🎮 Controls")
    c1, c2 = st.columns(2)
    c1.button(
        "⬅️ Prev",
        on_click=go_to,
        args=(st.session_state.idx - 1, total),
        use_container_width=True,
    )
    c2.button(
        "Next ➡️",
        on_click=go_to,
        args=(st.session_state.idx + 1, total),
        use_container_width=True,
    )

    st.divider()
    jump_val = st.number_input(f"Jump to (1-{total})", 1, total, value=st.session_state.idx + 1)
    if st.button("Go", use_container_width=True):
        go_to(jump_val - 1, total)
        st.rerun()

    st.divider()
    st.write("### 🤖 Inference")
    model_path = st.text_input("Model Path", value=DEFAULT_MODEL_PATH)
    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        if not Path(model_path).exists():
            st.error(f"Model file not found: {model_path}")
        else:
            with st.spinner("Processing..."):
                model = load_model_cached(model_path)
                st.session_state.pred_result = run_inference(model, img_rgb)
                st.rerun()

    st.write("#### 📊 Stats")
    st.write(f"**GT Count:** {gt_count}")
    if st.session_state.pred_result is not None:
        pred_c = st.session_state.pred_result[0]
        diff = pred_c - gt_count
        color = "red" if abs(diff) > ERROR_TOLERANCE else "green"
        st.write(f"**Pred Count:** {pred_c:.2f}")
        st.markdown(f"**Error:** :{color}[{diff:+.2f}]")


def main():
    st.set_page_config(layout="wide", page_title="Bird Annotation Viewer & Inference")
    init_session()

    data_root, split, pt_size = render_sidebar()
    images = list_images(data_root, split)
    if not images:
        st.error(f"❌ No images found in {data_root / 'images' / split}")
        st.stop()

    total = len(images)
    # Guard against the active index going out of range when the split changes.
    st.session_state.idx %= total

    curr = images[st.session_state.idx]
    img_bgr = cv2.imread(str(curr))
    if img_bgr is None:
        st.error(f"Failed to read image: {curr}")
        st.stop()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    points_px = load_points_px(curr, data_root, split, w, h)

    col1, col2 = st.columns([3, 1])
    with col1:
        render_image_panel(img_rgb, points_px, pt_size, curr.name, st.session_state.idx, total)
    with col2:
        render_controls(img_rgb, total, len(points_px))


main()
