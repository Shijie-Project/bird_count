import glob
import os
import warnings

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 尝试导入模型，确保项目结构正确
from models.shufflenet import get_shufflenet_density_model


warnings.filterwarnings("ignore")

# ================= 配置区域 =================
DEFAULT_DATA_ROOT = "./data/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

st.set_page_config(layout="wide", page_title="Bird Annotation Viewer & Inference")

# --- 预处理 (与训练保持一致) ---
# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
tf_normalize = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


def load_data(root_path, split):
    """加载指定 split 下的所有图片路径"""
    img_dir = os.path.join(root_path, "images", split)
    if not os.path.exists(img_dir):
        return []
    extensions = ["*.jpg", "*.png", "*.jpeg"]
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(img_files)


def load_annotation(img_path, root_path, split):
    """根据图片路径加载对应的 .txt 标注"""
    file_name = os.path.basename(img_path).rsplit(".", 1)[0]
    anno_path = os.path.join(root_path, "annotations", split, file_name + ".txt")
    points = None
    if os.path.exists(anno_path):
        try:
            data = np.loadtxt(anno_path, ndmin=2)
            if data.size > 0:
                points = data
        except Exception:
            pass
    return points, anno_path


@st.cache_resource
def load_model_cached(model_path):
    """加载模型并缓存，避免重复加载"""
    print(f"Loading model from {model_path}...")
    model = get_shufflenet_density_model()
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # 兼容 checkpoint 字典或直接 state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, img_rgb):
    """运行推理并返回 预测计数 和 热力图"""
    # 1. 预处理
    orig_h, orig_w = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)
    img_tensor = tf_normalize(pil_img).unsqueeze(0).to(DEVICE)

    # Padding 到 32 的倍数 (ShuffleNet要求)
    h, w = img_tensor.shape[2:]
    ph, pw = 0, 0
    if h % 32 != 0:
        ph = 32 - h % 32
    if w % 32 != 0:
        pw = 32 - w % 32
    if ph > 0 or pw > 0:
        img_tensor = F.pad(img_tensor, (0, pw, 0, ph))

    # 2. 推理
    with torch.no_grad():
        mu, _ = model(img_tensor)

    pred_count = torch.sum(mu).item()

    # 3. 生成热力图
    density_map = mu.squeeze().cpu().numpy()

    # 归一化 (0~1)
    if density_map.max() > 0:
        norm_density = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-6)
    else:
        norm_density = density_map

    # Resize 到原图大小
    norm_density = cv2.resize(norm_density, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # 转为 uint8
    norm_uint8 = (255 * norm_density).astype(np.uint8)

    # 应用 JET Colormap
    colormap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    # 4. 【关键修复】使用 mask 融合，去除蓝色背景
    threshold = 0.02  # 阈值，可根据效果微调 (0.01 ~ 0.1)
    mask = norm_density > threshold  # 只有大于阈值的区域才显示热力图

    overlay = img_rgb.copy()

    if mask.any():
        # 只在 mask 区域混合: 原图 0.6 + 热力图 0.4
        roi_orig = img_rgb[mask]
        roi_heat = colormap_rgb[mask]
        blended_roi = cv2.addWeighted(roi_orig, 0.6, roi_heat, 0.4, 0)
        overlay[mask] = blended_roi

    return pred_count, overlay


# --- Sidebar: 设置 ---
st.sidebar.title("🛠️ 设置")
data_root = st.sidebar.text_input("数据集根目录", value=DEFAULT_DATA_ROOT)
split = st.sidebar.selectbox("数据划分 (Split)", ["train", "val", "test"])

# --- 加载数据 ---
image_files = load_data(data_root, split)
total_images = len(image_files)

if total_images == 0:
    st.error(f"❌ 在 `{os.path.join(data_root, split, 'images')}` 下未找到图片！")
    st.stop()

# --- Session State ---
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "jump_val" not in st.session_state:
    st.session_state.jump_val = 1
# 存储预测结果，切换图片时清空
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None
if "last_processed_idx" not in st.session_state:
    st.session_state.last_processed_idx = -1

# 边界检查
if st.session_state.idx >= total_images:
    st.session_state.idx = total_images - 1
if st.session_state.idx < 0:
    st.session_state.idx = 0


# --- 状态更新函数 ---
def clear_pred():
    """切换图片时清除预测结果"""
    st.session_state.pred_result = None


def update_index(new_idx):
    st.session_state.idx = new_idx
    st.session_state.jump_val = new_idx + 1
    clear_pred()


def next_img():
    update_index((st.session_state.idx + 1) % total_images)


def prev_img():
    update_index((st.session_state.idx - 1) % total_images)


def search_img():
    query = st.session_state.search_query.strip()
    for i, path in enumerate(image_files):
        if query in os.path.basename(path):
            update_index(i)
            return
    st.toast(f"⚠️ 未找到 '{query}'", icon="🔍")


def jump_to_index():
    try:
        new_idx = st.session_state.jump_val - 1
        if 0 <= new_idx < total_images:
            st.session_state.idx = new_idx
            clear_pred()
        else:
            st.toast("⚠️ 索引越界", icon="❌")
            st.session_state.jump_val = st.session_state.idx + 1
    except Exception:
        pass


# --- 主界面 ---
col1, col2 = st.columns([3, 1])

# 获取当前图片
curr_img_path = image_files[st.session_state.idx]
curr_name = os.path.basename(curr_img_path)
points, anno_path = load_annotation(curr_img_path, data_root, split)

with col1:
    st.subheader(f"🖼️ {curr_name} ({st.session_state.idx + 1}/{total_images})")

    # 读取并显示 Ground Truth
    img_cv = cv2.imread(curr_img_path)
    if img_cv is not None:
        h, w, c = img_cv.shape
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 绘图逻辑
        disp_img = img_rgb.copy()
        point_radius = st.sidebar.slider("GT 点大小", 1, 10, 4)

        # 转换坐标
        gt_points_abs = points
        if points is not None and points.shape[1] == 3:  # 假设是 class, x, y 归一化格式
            gt_points_abs = points[:, 1:] * np.array([w, h])

        # 画点
        for p in gt_points_abs:
            cx, cy = int(p[0]), int(p[1])
            cv2.circle(disp_img, (cx, cy), point_radius, (255, 0, 0), -1)
            cv2.circle(disp_img, (cx, cy), point_radius + 2, (255, 255, 255), 1)

        # Tab 显示: 原始/GT vs 预测结果
        tab_gt, tab_pred = st.tabs(["📌 Ground Truth", "🔥 Model Prediction"])

        with tab_gt:
            st.image(disp_img, use_column_width=True, caption=f"GT Count: {len(points)}")

        with tab_pred:
            # 如果有预测结果，显示预测图
            if st.session_state.pred_result is not None:
                p_count, p_overlay = st.session_state.pred_result
                st.image(p_overlay, use_column_width=True, caption=f"Predicted Count: {p_count:.2f}")
            else:
                st.info("👈 请在右侧点击 '运行模型预测' 按钮查看结果")

with col2:
    st.write("### 🎮 控制台")
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 上一张", on_click=prev_img, use_container_width=True)
    with c2:
        st.button("下一张 ➡️", on_click=next_img, use_container_width=True)

    st.divider()
    st.write("### 🔍 导航")
    st.text_input("文件名搜索", key="search_query", on_change=search_img)
    st.number_input(f"跳转 ID (1-{total_images})", 1, total_images, key="jump_val", on_change=jump_to_index)

    st.divider()
    st.write("### 🤖 模型预测")

    # 模型路径输入
    model_path = st.text_input("模型路径 (.pth)", value="./ckpts/shufflenet_model_best.pth")

    # 运行按钮
    if st.button("🚀 运行模型预测", type="primary", use_container_width=True):
        if not os.path.exists(model_path):
            st.error(f"❌ 模型文件不存在: {model_path}")
        else:
            try:
                with st.spinner("正在加载模型并推理..."):
                    # 加载模型
                    model = load_model_cached(model_path)
                    # 运行推理
                    p_count, p_overlay = run_inference(model, img_rgb)
                    # 保存状态
                    st.session_state.pred_result = (p_count, p_overlay)
                    # 强制刷新界面以显示结果
                    st.rerun()
            except Exception as e:
                st.error(f"推理出错: {e}")

    # 显示数值对比
    st.write("#### 📊 统计")
    st.write(f"**GT 数量:** {len(points)}")
    if st.session_state.pred_result:
        pred_c = st.session_state.pred_result[0]
        diff = pred_c - len(points)
        color = "red" if abs(diff) > 5 else "green"
        st.write(f"**预测数量:** {pred_c:.2f}")
        st.markdown(f"**误差:** :{color}[{diff:+.2f}]")

    st.divider()
    st.caption(f"分辨率: {w}x{h}")
    st.caption(f"标注: {'✅ 存在' if os.path.exists(anno_path) else '❌ 缺失'}")
