import time

import torch
from torch.amp import autocast

from models.shufflenet import get_shufflenet_density_model


def benchmark(model, dummy_input, iterations=100, warmups=20, use_amp=False):
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(warmups):
            if use_amp:
                with autocast("cuda"):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            if use_amp:
                with autocast("cuda"):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_ms = (end_time - start_time) / iterations * 1000
    return avg_ms


def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available.")
        return

    # 输入尺寸参考你提供的 test.py: (21, 3, 720, 1080)
    # 注意：如果显存不足，请适当调小 batch size
    batch_size = 21
    dummy_input = torch.randn(batch_size, 3, 720, 1080).to(device)

    print(f"Benchmarking on {torch.cuda.get_device_name(0)}")
    print(f"Input Shape: {dummy_input.shape}\n")

    results = {}

    # 1. Baseline (Unfused, FP32)
    model_base = get_shufflenet_density_model(device=device, fuse=False)
    results["Baseline (FP32)"] = benchmark(model_base, dummy_input)
    print("Done: Baseline")

    # 2. Fused (FP32)
    model_fused = get_shufflenet_density_model(device=device, fuse=True)
    results["Fused (FP32)"] = benchmark(model_fused, dummy_input)
    print("Done: Fused")

    # 3. Fused + Autocast (AMP)
    results["Fused + Autocast"] = benchmark(model_fused, dummy_input, use_amp=True)
    print("Done: Fused + Autocast")

    # 4. Fused + Autocast + torch.compile (PyTorch 2.0+)
    # mode="reduce-overhead" 会使用 CUDA Graphs 减少内核启动开销
    try:
        if hasattr(torch, "compile"):
            model_compiled = torch.compile(model_fused)
            results["Fused + Autocast + Compile"] = benchmark(model_compiled, dummy_input, use_amp=True)
            print("Done: Fused + Autocast + Compile")
        else:
            print("Skipping torch.compile (Requires PyTorch 2.0+)")
    except Exception as e:
        print(f"torch.compile failed (this is common on some envs): {e}")

    # --- 打印对比表 ---
    print("\n" + "=" * 50)
    print(f"{'Method':<30} | {'Latency (ms)':<15}")
    print("-" * 50)

    baseline_time = results["Baseline (FP32)"]
    for name, latency in results.items():
        speedup = baseline_time / latency
        print(f"{name:<30} | {latency:>10.2f} ms ({speedup:.2f}x)")

    print("=" * 50)


if __name__ == "__main__":
    # 禁用一些不必要的调试开销
    torch.backends.cudnn.benchmark = True
    run_comparison()
