import argparse
import os
import warnings

import cv2
import dotenv
import numpy as np
import torch

from models.shufflenet import get_shufflenet_density_model


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)

CHECKPOINT_PATH = os.getenv("MODEL_PATH", None)
if CHECKPOINT_PATH is None:
    raise ValueError("Please set MODEL_PATH in .env file.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args():
    parser = argparse.ArgumentParser(description="Bird density video inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="input video path")
    parser.add_argument("--mask-image", type=str, default=None, help="mask image path; black pixels = masked region")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size for inference")
    parser.add_argument("--seconds", type=float, default=None, help="only process this many seconds of video")
    parser.add_argument("--device", default="0", help="cuda device index, e.g. 0")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    parser.add_argument(
        "--video-mode",
        default="split",
        choices=["pure", "overlay", "split"],
        help="output video type: overlay / pure density map / split side-by-side",
    )
    return parser.parse_args()


def preprocess_batch(frames_bgr, device, mean, std):
    """List of HxWx3 uint8 BGR frames -> normalized (B, 3, H, W) tensor on device."""
    arr = np.stack(frames_bgr, axis=0)  # (B, H, W, 3) uint8 BGR
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    t = t[..., [2, 1, 0]]  # BGR -> RGB
    t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    t.sub_(mean).div_(std)
    return t


def colorize_density(density_2d, target_w, target_h):
    """Density (H', W') float -> (H, W, 3) BGR colormap and the resized normalized map."""
    vmin = float(density_2d.min())
    vmax = float(density_2d.max())
    normed = (density_2d - vmin) / (vmax - vmin + 1e-5)
    normed = cv2.resize(normed, (target_w, target_h))
    heatmap = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap, normed


def density_color(pred_count):
    if pred_count < 100:
        return (0, 255, 0)  # green: low
    if pred_count < 110:
        return (0, 165, 255)  # orange: warn
    return (0, 0, 255)  # red: danger


def load_static_mask(mask_path, target_w, target_h):
    """Read mask_path as an image; return ((H, W) bool mask, contours list)."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_img is None:
        raise ValueError(f"Cannot open mask image: {mask_path}")

    mh, mw = mask_img.shape[:2]
    if mw != target_w or mh != target_h:
        raise ValueError(f"Mask image size ({mw}x{mh}) != target video size ({target_w}x{target_h}).")

    bool_mask = np.all(mask_img < 10, axis=2)
    contours, _ = cv2.findContours((~bool_mask).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return bool_mask, contours


def run_video(args, model, device):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.seconds is not None:
        cap_frames = max(1, int(args.seconds * fps))
        total_frames = min(total_frames, cap_frames) if total_frames > 0 else cap_frames

    black_mask = None
    mask_contours = None
    if args.mask_image is not None:
        black_mask, mask_contours = load_static_mask(args.mask_image, w, h)

    base, _ = os.path.splitext(args.input)
    suffix = {"overlay": "_overlay", "pure": "_pure", "split": "_split"}[args.video_mode]
    out_path = base + suffix + ".mp4"

    out_w = w * 2 if args.video_mode == "split" else w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing: {out_path}")

    print(f"Video: {args.input}  ({w}x{h} @ {fps:.1f}fps, {total_frames} frames)")
    print(f"Mask image: {args.mask_image}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {out_path}  mode={args.video_mode}")

    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    use_amp = (not args.no_amp) and device.type == "cuda"

    frame_buffer = []
    orig_buffer = []
    pred_counts = []
    frame_idx = 0
    log_every = max(int(round(fps)), 1)

    def flush():
        if not frame_buffer:
            return

        inputs = preprocess_batch(frame_buffer, device, mean, std)

        amp_ctx = torch.autocast(device_type=device.type, enabled=use_amp)
        with torch.inference_mode(), amp_ctx:
            outputs = model(inputs)  # (B, 1, H', W')

        # Single GPU->CPU sync per batch instead of one per frame.
        out_fp32 = outputs.float()
        sums = out_fp32.sum(dim=(1, 2, 3)).cpu().numpy()
        maps = out_fp32[:, 0].cpu().numpy()

        for j in range(len(frame_buffer)):
            pred_count = float(sums[j])
            pred_counts.append(pred_count)
            orig_frame = orig_buffer[j]

            if args.video_mode == "pure":
                heatmap, _ = colorize_density(maps[j], w, h)
                writer.write(heatmap)
                continue

            if args.video_mode == "split":
                heatmap, _ = colorize_density(maps[j], w, h)
                writer.write(np.concatenate([orig_frame, heatmap], axis=1))
                continue

            # overlay mode
            heatmap, normed = colorize_density(maps[j], w, h)
            overlay_frame = orig_frame.copy()
            heat_mask = normed > 0.01
            if black_mask is not None:
                heat_mask &= ~black_mask
            if heat_mask.any():
                overlay_frame[heat_mask] = cv2.addWeighted(orig_frame[heat_mask], 0.5, heatmap[heat_mask], 0.5, 0)
            if mask_contours is not None:
                cv2.drawContours(overlay_frame, mask_contours, -1, density_color(pred_count), thickness=3)
            cv2.putText(
                overlay_frame,
                f"Est: {pred_count:.1f}",
                (55, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            writer.write(overlay_frame)

        frame_buffer.clear()
        orig_buffer.clear()

    try:
        while True:
            if total_frames > 0 and frame_idx >= total_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if black_mask is not None:
                orig_frame = frame.copy()
                frame[black_mask] = 0
            else:
                orig_frame = frame

            frame_buffer.append(frame)
            orig_buffer.append(orig_frame)

            if len(frame_buffer) >= args.batch_size:
                flush()

            frame_idx += 1
            if frame_idx % log_every == 0 or frame_idx == total_frames:
                print(f"  [{frame_idx}/{total_frames}]")

        flush()
    finally:
        cap.release()
        writer.release()

    print(f"\nDone. Frames processed: {frame_idx}")
    if pred_counts:
        arr = np.asarray(pred_counts)
        print(f"Pred count — Mean: {arr.mean():.2f}, Min: {arr.min():.2f}, Max: {arr.max():.2f}")
    else:
        print("No frames were processed.")
    print(f"Saved to: {out_path}")


def main():
    args = get_args()

    device = DEVICE
    if device.type == "cuda":
        try:
            torch.cuda.set_device(int(args.device))
        except (ValueError, RuntimeError) as e:
            print(f"Warning: failed to set CUDA device to {args.device!r}: {e}")
        torch.backends.cudnn.benchmark = True

    model = get_shufflenet_density_model(model_path=CHECKPOINT_PATH, device=device, fuse=True)
    model.eval()

    run_video(args, model, device)


if __name__ == "__main__":
    main()
