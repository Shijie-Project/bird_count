import argparse
import os
import warnings

import dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.bird import Bird
from models.shufflenet import get_shufflenet_density_model


dotenv.load_dotenv()

CHECKPOINT_PATH = os.getenv("MODEL_PATH", None)
if CHECKPOINT_PATH is None:
    raise ValueError("Please set MODEL_PATH in .env file.")


warnings.simplefilter("ignore", UserWarning)


def get_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument("--split", default="val", choices=["val", "train"], help="split dataset")
    parser.add_argument("--data-path", type=str, default="../data", help="saved data path")
    parser.add_argument("--no-density-map", action="store_true", help="no density map")
    parser.add_argument("--video", type=str, default=None, help="input video path (enables video mode)")
    parser.add_argument("--video-output", type=str, default=None, help="output video path (default: <input>_out.mp4)")
    parser.add_argument(
        "--video-mode",
        default="both",
        choices=["overlay", "pure", "both"],
        help="output video type: overlay / pure density map / both side-by-side",
    )
    # ----------------
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# 将单帧 numpy BGR 图像 (H, W, 3) 经模型推理后渲染成可视化帧
# 返回: (overlay_frame, pure_frame, pred_count)
# ---------------------------------------------------------------------------
def process_frame(frame_bgr, model, device):
    import cv2
    import torchvision.transforms.functional as TF
    from PIL import Image

    h, w = frame_bgr.shape[:2]

    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = TF.to_tensor(img_pil)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inputs = tensor.unsqueeze(0).to(device).float()

    with torch.set_grad_enabled(False):
        outputs = model(inputs)

    pred_count = outputs.sum().item()

    # --- 生成 density heatmap ---
    vis_map = outputs[0, 0].cpu().numpy()
    normed_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
    normed_map = cv2.resize(normed_map, (w, h))
    vis_uint8 = (normed_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(vis_uint8, cv2.COLORMAP_JET)

    # pure：直接返回 heatmap
    pure_frame = heatmap.copy()

    # overlay：只在高密度区域叠加
    overlay_frame = frame_bgr.copy()
    mask = normed_map > 0.01
    if mask.any():
        blended = cv2.addWeighted(frame_bgr[mask], 0.5, heatmap[mask], 0.5, 0)
        overlay_frame[mask] = blended

    label = f"Est: {pred_count:.1f}"
    cv2.putText(overlay_frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay_frame, pure_frame, pred_count


def run_video(args, model, device):
    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 决定输出路径
    if args.video_output:
        out_path = args.video_output
    else:
        base, ext = os.path.splitext(args.video)
        suffix = {"overlay": "_overlay", "pure": "_pure", "both": "_both"}[args.video_mode]
        out_path = base + suffix + (ext if ext else ".mp4")

    out_w = w * 2 if args.video_mode == "both" else w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, h))

    print(f"Video: {args.video}  ({w}x{h} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Output: {out_path}  mode={args.video_mode}")

    frame_idx = 0
    pred_counts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay_frame, pure_frame, pred_count = process_frame(frame, model, device)
        pred_counts.append(pred_count)

        if args.video_mode == "overlay":
            out_frame = overlay_frame
        elif args.video_mode == "pure":
            out_frame = pure_frame
        else:  # both：左 overlay，右 pure density map
            out_frame = np.concatenate([overlay_frame, pure_frame], axis=1)

        writer.write(out_frame)

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  [{frame_idx}/{total_frames}] pred={pred_count:.1f}")

    cap.release()
    writer.release()

    print(f"\nDone. Frames processed: {frame_idx}")
    print(
        f"Pred count — Mean: {np.mean(pred_counts):.2f}, Min: {np.min(pred_counts):.2f}, Max: {np.max(pred_counts):.2f}"
    )
    print(f"Saved to: {out_path}")


def main():
    args = get_args()

    torch.cuda.set_device(int(args.device))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_shufflenet_density_model()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # ===== 视频模式 =====
    if args.video is not None:
        run_video(args, model, device)
        return
    # ====================

    dataset = Bird(args.data_path, args.crop_size, 8, split=args.split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    density_map_path = None
    if not args.no_density_map:
        import cv2

        density_map_path = os.path.join(
            os.path.join(args.data_path, "density_maps"), CHECKPOINT_PATH.split("/")[-1].split(".")[0]
        )
        if not os.path.exists(density_map_path):
            os.makedirs(density_map_path)

    image_errs = []
    ground_truths = []
    accuracies = []

    for img, inputs, gt_discrete, name in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        count = gt_discrete[0].sum().item()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        pred_count = outputs.sum().item()
        img_err = count - pred_count

        # --- 计算单张图片的 Accuracy ---
        if count > 0:
            rel_err = abs(img_err) / count
            acc_i = max(0, 1.0 - rel_err)
        else:
            acc_i = 1.0 if abs(pred_count) < 0.5 else 0.0

        accuracies.append(acc_i)

        print(f"{name[0]}: GT {count:.1f}, Pred {pred_count:.1f}, Err {img_err:.1f}, Acc {acc_i:.4f}")

        image_errs.append(img_err)
        ground_truths.append(count)

        if density_map_path is not None:
            import cv2

            original_img = cv2.imread(img[0])
            h, w = original_img.shape[:2]

            vis_map = outputs[0, 0].cpu().numpy()

            normed_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
            normed_map = cv2.resize(normed_map, (w, h))

            vis_img_uint8 = (normed_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(vis_img_uint8, cv2.COLORMAP_JET)

            # --- 纯净 density map ---
            pure_save_path = os.path.join(density_map_path, str(name[0]) + "_density_pure.png")
            cv2.imwrite(pure_save_path, heatmap)
            # ------------------------

            overlay_img = original_img.copy()
            alpha = 0.5
            beta = 0.5
            threshold = 0.01
            mask = normed_map > threshold

            if mask.any():
                roi_orig = original_img[mask]
                roi_heat = heatmap[mask]
                blended_roi = cv2.addWeighted(roi_orig, alpha, roi_heat, beta, 0)
                overlay_img[mask] = blended_roi

            cv2.putText(
                overlay_img,
                f"gt:{count:.1f}, est:{outputs.sum().item():.1f}, diff:{img_err:.1f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            save_path = os.path.join(density_map_path, str(name[0]) + "_density.png")
            cv2.imwrite(save_path, overlay_img)

    image_errs = np.array(image_errs)
    accuracies = np.array(accuracies)

    # 传统的全局指标
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    average_cnt = np.mean(ground_truths)
    global_acc = 1 - mae / average_cnt

    # Per-image Accuracy 的统计
    acc_mean = np.mean(accuracies)
    acc_var = np.var(accuracies)
    acc_std = np.std(accuracies)

    print(f"\nTotal Instance: {sum(ground_truths)}")
    print(f"Global Metrics: MAE {mae:.4f}, MSE {mse:.4f}, Global Acc {global_acc:.4f}")
    print("-" * 40)
    print("Per-image Accuracy Stats:")
    print(f"  Mean (Avg Acc): {acc_mean:.4f}")
    print(f"  Variance      : {acc_var:.4f}")
    print(f"  Std Deviation : {acc_std:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
