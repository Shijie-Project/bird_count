import argparse
import os
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import Bird
from models import VGG19, ShuffleNetV2_x1_0


warnings.simplefilter("ignore", UserWarning)


def get_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument("--split", default="val", choices=["val", "train"], help="split dataset")
    parser.add_argument("--model", default="shufflenet", choices=["vgg", "shufflenet"], help="model name")
    parser.add_argument("--checkpoint", type=str, default="./ckpts/shufflenet_model_best.pth", help="saved model path")
    parser.add_argument("--data-path", type=str, default="./data/bird_count/", help="saved model path")
    parser.add_argument("--crop-size", type=int, default=512, help="the crop size of the train image")
    parser.add_argument(
        "--density-map-path", type=str, default="./density_maps", help="folder to save predicted density maps."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    torch.cuda.set_device(int(args.device))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Bird(args.data_path, args.crop_size, 8, split=args.split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    density_map_path = None
    if args.density_map_path:
        import cv2

        density_map_path = os.path.join(args.density_map_path, args.model)
        if not os.path.exists(density_map_path):
            os.makedirs(density_map_path)

    if args.model == "vgg":
        model = VGG19()
    elif args.model == "shufflenet":
        model = ShuffleNetV2_x1_0()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    image_errs = []
    ground_truths = []
    for img, inputs, count, name in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        count = count[0].item()

        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)

        img_err = count - outputs.sum().item()

        print(name, img_err, count, outputs.sum().item())
        image_errs.append(img_err)
        ground_truths.append(count)

        if density_map_path is not None:
            original_img = cv2.imread(img[0])
            h, w = original_img.shape[:2]

            vis_map = outputs[0, 0].cpu().numpy()

            # 归一化到 0~1 (这是关键，这个值将作为透明度 Alpha)
            # norm_map 的范围严格控制在 0.0 到 1.0 之间
            normed_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)

            # Resize 密度图到原图大小
            normed_map = cv2.resize(normed_map, (w, h))

            vis_img_uint8 = (normed_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(vis_img_uint8, cv2.COLORMAP_JET)

            # 初始化最终结果为原图 (这样背景就完全是原图，没有蓝色)
            overlay_img = original_img.copy()

            # 【局部融合】
            # 只在 mask 为 True 的区域进行加权融合
            # alpha=0.6 (原图权重), beta=0.4 (热力图权重)
            # 你可以把 beta 调高到 0.7 甚至 1.0 让“点”更亮更明显
            alpha = 0.5
            beta = 0.5

            threshold = 0.01

            # 创建掩码 (Boolean Mask)
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
                1.2,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            save_path = os.path.join(density_map_path, str(name[0]) + "_density.png")
            cv2.imwrite(save_path, overlay_img)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    average_cnt = np.mean(ground_truths)
    acc = 1 - mae / average_cnt
    print(f"Total Instance: {sum(ground_truths)}")
    print(f"{args.model}: mae {mae}, mse {mse}, acc {acc}\n")


if __name__ == "__main__":
    main()
