import os
import os.path as osp

import numpy as np


splits = ["train", "val"]

for split in splits:
    anno_dir = osp.join(split, "anno_cnt")

    anno_files = os.listdir(anno_dir)
    anno_files.sort()

    all_counts = []
    for name in anno_files:
        keypoints = np.loadtxt(osp.join(anno_dir, name), ndmin=2)
        all_counts.append(keypoints.shape[0])
    count_sum = sum(all_counts)
    count_len = len(all_counts)
    print(
        f"{split.upper()} -\t"
        f"all:{count_sum}, "
        f"min:{min(all_counts)}, "
        f"max:{max(all_counts)}, "
        f"avg:{count_sum / count_len:.1f}, "
        f"img:{count_len}"
    )
