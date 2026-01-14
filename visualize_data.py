import glob
import os
import warnings

import cv2
import numpy as np
import streamlit as st


warnings.filterwarnings("ignore")

# ================= 配置区域 =================
# 默认数据路径 (你可以修改这里，或者在网页侧边栏修改)
DEFAULT_DATA_ROOT = "./data/"
# ===========================================

st.set_page_config(layout="wide", page_title="Bird Annotation Viewer")


def load_data(root_path, split):
    """加载指定 split 下的所有图片路径"""
    img_dir = os.path.join(root_path, "images", split)
    if not os.path.exists(img_dir):
        return []

    # 支持多种后缀
    extensions = ["*.jpg", "*.png", "*.jpeg"]
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(img_dir, ext)))

    return sorted(img_files)


def load_annotation(img_path, root_path, split):
    """根据图片路径加载对应的 .txt 标注"""
    file_name = os.path.basename(img_path).rsplit(".", 1)[0]
    anno_path = os.path.join(root_path, "annotations", split, file_name + ".txt")

    points = []
    if os.path.exists(anno_path):
        try:
            # 假设格式是: x y (每行一个点)
            data = np.loadtxt(anno_path, ndmin=2)
            if data.size > 0:
                points = data
        except Exception:
            pass
    return points, anno_path


# --- Sidebar: 设置 ---
st.sidebar.title("🛠️ 设置")
data_root = st.sidebar.text_input("数据集根目录", value=DEFAULT_DATA_ROOT)
split = st.sidebar.selectbox("数据划分 (Split)", ["train", "val", "test"])

# --- 加载数据 ---
image_files = load_data(data_root, split)
total_images = len(image_files)

if total_images == 0:
    st.error(f"❌ 在 `{os.path.join(data_root, split, 'images')}` 下未找到图片！请检查路径。")
    st.stop()

# --- Session State: 管理当前查看的索引 ---
if "idx" not in st.session_state:
    st.session_state.idx = 0

# 确保索引不越界
if st.session_state.idx >= total_images:
    st.session_state.idx = total_images - 1
if st.session_state.idx < 0:
    st.session_state.idx = 0


# --- 功能函数 ---
def next_img():
    st.session_state.idx = (st.session_state.idx + 1) % total_images


def prev_img():
    st.session_state.idx = (st.session_state.idx - 1) % total_images


def search_img():
    query = st.session_state.search_query.strip()
    found = False
    for i, path in enumerate(image_files):
        if query in os.path.basename(path):
            st.session_state.idx = i
            found = True
            break
    if not found:
        st.toast(f"⚠️ 未找到包含 '{query}' 的图片", icon="🔍")


# --- 主界面布局 ---
col1, col2 = st.columns([3, 1])

with col1:
    # 获取当前图片信息
    curr_img_path = image_files[st.session_state.idx]
    curr_name = os.path.basename(curr_img_path)
    points, anno_path = load_annotation(curr_img_path, data_root, split)

    st.subheader(f"🖼️ {curr_name} ({st.session_state.idx + 1}/{total_images})")

    # --- 绘图逻辑 ---
    # 读取图片 (OpenCV format)
    img_cv = cv2.imread(curr_img_path)
    h, w, c = img_cv.shape

    if points.shape[1] == 3:
        points = points[:, 1:]
        points = points * np.array([w, h])

    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 绘制点
    point_radius = st.sidebar.slider("点的大小", 1, 10, 4)
    point_color = (255, 0, 0)  # 红色

    for p in points:
        x, y = round(p[0]), round(p[1])
        cv2.circle(img_cv, (x, y), point_radius, point_color, -1)  # 实心点
        # 可选：画个圈强调
        cv2.circle(img_cv, (x, y), point_radius + 2, (255, 255, 255), 1)

        # 显示图片
    st.image(img_cv, use_column_width=True)

with col2:
    st.write("### 🎮 控制台")

    # 导航按钮
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 上一张", on_click=prev_img, use_container_width=True)
    with c2:
        st.button("下一张 ➡️", on_click=next_img, use_container_width=True)

    st.divider()

    # 搜索功能
    st.write("### 🔍 搜索")
    st.text_input("输入文件名 (例如: img_005)", key="search_query", on_change=search_img)

    st.divider()

    # 信息面板
    st.write("### 📊 信息")
    st.info(f"**鸡的数量 (Count):** {len(points)}")
    st.text(f"分辨率: {img_cv.shape[1]} x {img_cv.shape[0]}")

    st.write("标注文件路径:")
    if os.path.exists(anno_path):
        st.success(f"`{os.path.basename(anno_path)}` (存在)")
    else:
        st.error(f"`{os.path.basename(anno_path)}` (缺失)")

    # 原始坐标展示 (Debug用)
    with st.expander("查看原始坐标数据"):
        st.write(points)

# --- 键盘快捷键提示 ---
st.sidebar.markdown("---")
st.sidebar.markdown("**提示:**")
st.sidebar.markdown("- 确保你的数据结构符合 `Split/images` 和 `Split/annotations`")
st.sidebar.markdown("- 标注文件格式应为 `.txt`，每行 `x y`")
