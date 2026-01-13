import argparse
import os
import warnings

import torch

from trainer import Trainer
from utils import set_seed


warnings.simplefilter("ignore", UserWarning)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Train")
    # parser.add_argument('--data-dir', default='D:/data/bird_count/split', help='data path')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data-dir", default="./data", help="data path")
    parser.add_argument("--lr", type=float, default=1e-5, help="the initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="the weight decay")
    parser.add_argument("--resume", default="", type=str, help="the path of resume training model")
    parser.add_argument("--max-epoch", type=int, default=1000, help="max training epoch")
    parser.add_argument("--val-epoch", type=int, default=5, help="the num of steps to log training information")
    parser.add_argument("--val-start", type=int, default=50, help="the epoch start to val")
    parser.add_argument("--batch-size", type=int, default=8, help="train batch size")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument("--num-workers", type=int, default=8, help="the num of training process")
    parser.add_argument("--crop-size", type=int, default=512, help="the crop size of the train image")
    parser.add_argument("--wot", type=float, default=0.1, help="weight on OT loss")
    parser.add_argument("--wtv", type=float, default=0.01, help="weight on TV loss")
    parser.add_argument("--reg", type=float, default=10.0, help="entropy regularization in sinkhorn")
    parser.add_argument("--num-of-iter-in-ot", type=int, default=100, help="sinkhorn iterations")
    parser.add_argument("--norm-cood", type=int, default=0, help="whether to norm cood when computing distance")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
