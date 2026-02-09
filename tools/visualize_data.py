import glob
import os
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


# Add current directory to path to fix "No module named models"
sys.path.append(os.path.dirname(Path(__file__).parents[1].as_posix()))


try:
    from models import get_model
except ImportError:
    st.error("❌ Could not find 'models.py'. Ensure it's in the same folder as this script.")
    st.stop()

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
DEFAULT_DATA_ROOT = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

st.set_page_config(layout="wide", page_title="Bird Annotation Viewer & Inference")

# --- 预处理 (与训练保持一致) ---
# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
tf_normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_data(root_path, split):
    """Load all image paths for a specific data split."""
    img_dir = os.path.join(root_path, "images", split)
    if not os.path.exists(img_dir):
        return []
    # Support multiple extensions if needed (.jpg, .png)
    img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    return sorted(img_files)


def load_annotation(img_path, root_path, split):
    """Load .txt annotations corresponding to the image path."""
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    anno_path = os.path.join(root_path, "annotations", split, file_name + ".txt")
    points = np.empty((0, 2))

    if os.path.exists(anno_path):
        try:
            # ndmin=2 ensures data is always 2D even with one point
            data = np.loadtxt(anno_path, ndmin=2)
            if data.size > 0:
                points = data
        except Exception as e:
            st.warning(f"Failed to load annotation: {e}")
    return points, anno_path


@st.cache_resource
def load_model_cached(model_path):
    """Load and cache the model to prevent reloading on every interaction."""
    model = get_model(model_path, device=DEVICE, fuse=True)
    model.eval()
    return model


def run_inference(model, img_rgb):
    """Run inference and return predicted count and overlay heatmap."""
    orig_h, orig_w = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)
    img_tensor = tf_normalize(pil_img).unsqueeze(0).to(DEVICE)

    # Padding to multiples of 32 (Requirement for many CNN architectures like ShuffleNet)
    h, w = img_tensor.shape[2:]
    ph = (32 - h % 32) % 32
    pw = (32 - w % 32) % 32
    if ph > 0 or pw > 0:
        img_tensor = F.pad(img_tensor, (0, pw, 0, ph))

    with torch.no_grad():
        mu, _ = model(img_tensor)

    # Result logic
    pred_count = torch.sum(mu).item()
    density_map = mu.squeeze().cpu().numpy()

    # Normalization (0~1) for visualization
    d_max = density_map.max()
    norm_density = (density_map - density_map.min()) / (d_max - density_map.min() + 1e-6) if d_max > 0 else density_map

    # Resize heatmap back to original image dimensions
    norm_density = cv2.resize(norm_density, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Generate Heatmap Overlay
    norm_uint8 = (255 * norm_density).astype(np.uint8)
    colormap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    # Use a threshold mask to prevent "blue wash" over empty areas
    threshold = 0.05
    mask = norm_density > threshold
    overlay = img_rgb.copy()

    if mask.any():
        # Alpha blending: 60% original image, 40% heatmap
        overlay[mask] = cv2.addWeighted(img_rgb, 0.6, colormap_rgb, 0.4, 0)[mask]

    return pred_count, overlay


# --- Sidebar: 设置 ---
st.sidebar.title("🛠️ Settings")
data_root = st.sidebar.text_input("Dataset Root", value=DEFAULT_DATA_ROOT)

# Dynamic split selection based on directory structure
if os.path.exists(os.path.join(data_root, "images")):
    available_splits = os.listdir(os.path.join(data_root, "images"))
    split = st.sidebar.selectbox("Data Split", available_splits)
else:
    st.error(f"Path not found: {os.path.join(data_root, 'images')}")
    st.stop()

# --- Data Loading ---
image_files = load_data(data_root, split)
total_images = len(image_files)

if total_images == 0:
    st.error(f"❌ No images found in `{os.path.join(data_root, 'images', split)}`!")
    st.stop()


# --- Session State Management ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None


def update_index(new_idx):
    st.session_state.idx = new_idx % total_images
    st.session_state.pred_result = None  # Clear prediction when image changes


def next_img():
    update_index(st.session_state.idx + 1)


def prev_img():
    update_index(st.session_state.idx - 1)


col1, col2 = st.columns([3, 1])

curr_img_path = image_files[st.session_state.idx]
curr_name = os.path.basename(curr_img_path)
points, anno_path = load_annotation(curr_img_path, data_root, split)

with col1:
    st.subheader(f"🖼️ {curr_name} ({st.session_state.idx + 1}/{total_images})")

    img_cv = cv2.imread(curr_img_path)
    if img_cv is not None:
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Ground Truth Visualization
        disp_img = img_rgb.copy()
        pt_size = st.sidebar.slider("GT Point Size", 1, 10, 4)

        for p in points:
            # Handle normalized or pixel coordinates
            cx, cy = (int(p[0] * w), int(p[1] * h)) if points.shape[1] == 3 else (int(p[0]), int(p[1]))
            cv2.circle(disp_img, (cx, cy), pt_size, (255, 0, 0), -1)
            cv2.circle(disp_img, (cx, cy), pt_size + 2, (255, 255, 255), 1)

        tab_gt, tab_pred = st.tabs(["📌 Ground Truth", "🔥 Model Prediction"])
        with tab_gt:
            st.image(disp_img, width="stretch", caption=f"GT Count: {len(points)}")
        with tab_pred:
            if st.session_state.pred_result:
                p_count, p_overlay = st.session_state.pred_result
                st.image(p_overlay, width="stretch", caption=f"Predicted Count: {p_count:.2f}")
            else:
                st.info("Run model prediction from the sidebar to view results.")

with col2:
    st.write("### 🎮 Controls")
    c1, c2 = st.columns(2)
    c1.button("⬅️ Prev", on_click=prev_img, use_container_width=True)
    c2.button("Next ➡️", on_click=next_img, use_container_width=True)

    st.divider()
    # Jump to specific ID
    jump_val = st.number_input(f"Jump to (1-{total_images})", 1, total_images, value=st.session_state.idx + 1)
    if st.button("Go", use_container_width=True):
        update_index(jump_val - 1)
        st.rerun()

    st.divider()
    st.write("### 🤖 Inference")
    model_path = st.text_input("Model Path", value="./ckpts/shufflenet_model_best.pth")

    if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
        if os.path.exists(model_path):
            with st.spinner("Processing..."):
                model = load_model_cached(model_path)
                st.session_state.pred_result = run_inference(model, img_rgb)
                st.rerun()
        else:
            st.error("Model file not found.")

    st.write("#### 📊 Stats")
    st.write(f"**GT Count:** {len(points)}")
    if st.session_state.pred_result:
        pred_c = st.session_state.pred_result[0]
        st.write(f"**Pred Count:** {pred_c:.2f}")
        diff = pred_c - len(points)
        st.markdown(f"**Error:** :{'red' if abs(diff) > 5 else 'green'}[{diff:+.2f}]")
