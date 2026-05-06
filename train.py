"""Training entrypoint for the chicken-density model."""

import argparse
import os
import warnings

from trainer import Trainer
from utils import set_seed


warnings.simplefilter("ignore", UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ShuffleNet density model")

    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default="../data", help="dataset root")
    g.add_argument("--checkpoint-dir", default="../ckpts", help="where to write checkpoints")
    g.add_argument("--crop-size", type=int, default=512, help="train crop size (must be divisible by 8)")
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--num-workers", type=int, default=8)

    g = p.add_argument_group("optimization")
    g.add_argument("--lr", type=float, default=1e-5, help="peak learning rate (cosine target)")
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--max-epoch", type=int, default=1000)
    g.add_argument("--warmup-epochs", type=int, default=5, help="linear warmup before cosine annealing")
    g.add_argument("--no-scheduler", action="store_true", help="disable LR scheduler entirely")
    g.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for the eval-time copy")
    g.add_argument(
        "--no-freeze-backbone-bn",
        action="store_true",
        help="train backbone BN running stats (default: frozen to ImageNet stats)",
    )

    g = p.add_argument_group("loss (DM-Count + auxiliary)")
    g.add_argument("--wot", type=float, default=0.1, help="OT loss weight")
    g.add_argument("--wtv", type=float, default=0.01, help="DM-Count distribution-match (TV) weight")
    g.add_argument("--wcount", type=float, default=1.0, help="global count L1 weight")
    g.add_argument("--waux", type=float, default=1.0, help="Gaussian-density aux loss weight")
    g.add_argument(
        "--aux-sigma", type=float, default=2.0, help="Gaussian sigma in density-map pixels (0 disables aux loss)"
    )
    g.add_argument(
        "--dense-weight-alpha",
        type=float,
        default=0.0,
        help="upweight aux pixels by (1 + alpha * GT_density); 0 = uniform",
    )
    g.add_argument("--reg", type=float, default=10.0, help="entropy regularization in Sinkhorn")
    g.add_argument("--num-of-iter-in-ot", type=int, default=100, help="Sinkhorn iterations")
    g.add_argument("--norm-cood", action="store_true", help="normalize coords in OT distance computation")

    g = p.add_argument_group("evaluation")
    g.add_argument("--val-epoch", type=int, default=5, help="run validation every N epochs")
    g.add_argument("--val-start", type=int, default=50, help="first epoch eligible for validation")

    g = p.add_argument_group("misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument(
        "--deterministic",
        action="store_true",
        help="enable cuDNN deterministic mode (slower; for fully reproducible runs)",
    )
    g.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value (e.g. '0')")
    g.add_argument("--resume", default="", help="checkpoint to resume from (.tar or .pth)")

    return p.parse_args()


def main():
    args = parse_args()
    # Restrict visible GPUs before any CUDA initialization. setdefault so a
    # value already set in the shell takes precedence over the CLI default.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())

    set_seed(args.seed, deterministic=args.deterministic)
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
