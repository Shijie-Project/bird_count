import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.bird import Bird
from models.shufflenet import get_shufflenet_density_model


warnings.simplefilter("ignore", UserWarning)


def get_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument("--split", default="val", choices=["val", "train"], help="split dataset")
    parser.add_argument("--legacy-data", action="store_true", help="whether to use legacy data format")
    parser.add_argument("--model", default="shufflenet", choices=["vgg", "shufflenet"], help="model name")
    parser.add_argument("--checkpoint", type=str, default="./ckpts/best_model.pth", help="saved model path")
    parser.add_argument("--data-path", type=str, default="./data", help="saved model path")
    parser.add_argument("--crop-size", type=int, default=512, help="the crop size of the train image")
    parser.add_argument(
        "--density-map-path", type=str, default="./data/density_maps", help="folder to save predicted density maps."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    torch.cuda.set_device(int(args.device))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Bird(args.data_path, args.crop_size, 8, split=args.split + ("_legacy" if args.legacy_data else ""))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    density_map_path = None
    if args.density_map_path:
        import cv2

        density_map_path = os.path.join(args.density_map_path, args.checkpoint.split("/")[-1].split(".")[0])
        if not os.path.exists(density_map_path):
            os.makedirs(density_map_path)

    model = get_shufflenet_density_model()

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    image_errs = []
    ground_truths = []
    accuracies = []  # 新增：用于存储每张图片的准确率

    for img, inputs, gt_discrete, name in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        count = gt_discrete[0].sum().item()

        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)

        pred_count = outputs.sum().item()
        img_err = count - pred_count

        # --- 计算单张图片的 Accuracy ---
        # 定义: Acc = 1 - |Err| / GT
        # 范围控制在 [0, 1] 之间，防止负数影响方差统计
        if count > 0:
            rel_err = abs(img_err) / count
            acc_i = max(0, 1.0 - rel_err)
        else:
            # 如果 GT 为 0，预测值小于 0.5 算全对(1.0)，否则算全错(0.0)
            acc_i = 1.0 if abs(pred_count) < 0.5 else 0.0

        accuracies.append(acc_i)
        # -----------------------------

        print(f"{name[0]}: GT {count:.1f}, Pred {pred_count:.1f}, Err {img_err:.1f}, Acc {acc_i:.4f}")

        image_errs.append(img_err)
        ground_truths.append(count)

        if density_map_path is not None:
            original_img = cv2.imread(img[0])
            h, w = original_img.shape[:2]

            vis_map = outputs[0, 0].cpu().numpy()

            normed_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
            normed_map = cv2.resize(normed_map, (w, h))

            vis_img_uint8 = (normed_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(vis_img_uint8, cv2.COLORMAP_JET)

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

    # 新增：Per-image Accuracy 的统计
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
