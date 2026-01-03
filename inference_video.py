import logging
import math
import queue
import time
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

# Import your models
from models import VGG19, ShuffleNetV2_x1_0
from utils import SimulatedCamera


# --- Configuration ---
@dataclass
class Config:
    # Path to the test video file
    VIDEO_PATH: str = r"rtsp://127.0.0.1:8554/live/test"

    # Checkpoints
    CKPT_VGG: str = "./ckpts/vgg_model_best.pth"
    CKPT_SHUFFLE: str = "./ckpts/shufflenet_model_best.pth"

    # System Settings
    NUM_STREAMS: int = 1  # 这里的路数
    BATCH_SIZE: int = 1  # 必须匹配 NUM_STREAMS
    NUM_POST_PROCESSES: int = 1  # 后处理进程数 (建议设为 2-4)
    DEVICE_ID: int = 0
    USE_VGG: bool = False

    # Simulation Settings
    TARGET_FPS: int = 10

    # Queue Sizes
    FRAME_QUEUE_SIZE: int = 100
    RESULT_QUEUE_SIZE: int = 100
    MONITOR_QUEUE_SIZE: int = 50

    # Input Resolution
    INPUT_SIZE: tuple[int, int] = (640, 360)

    # Monitor Settings
    ENABLE_MONITOR: bool = True
    VIS_INTERVAL: int = 1  # 每多少帧画一次热力图 (设为 1 用于压力测试，生产环境建议 2)

    # Debug Settings
    DEBUG: bool = True


cfg = Config()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.DEBUG if cfg.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InferenceEngine")
warnings.filterwarnings("ignore")


# --- Helper: Generate Grid Layout ---
def create_mosaic_layout(num_streams, frame_size):
    """Calculates the optimal grid size (rows, cols) for the monitor."""
    cols = math.ceil(math.sqrt(num_streams))
    rows = math.ceil(num_streams / cols)

    # Calculate single tile size (Small thumbnail to fit screen)
    # Assuming 1920x1080 Full HD Monitor
    screen_w, screen_h = 1920, 1080  # noqa: F841
    tile_w = screen_w // cols
    tile_h = int(tile_w * (frame_size[1] / frame_size[0]))  # Keep aspect ratio

    return rows, cols, (tile_w, tile_h)


# --- 1. Frame Stream ---
def frame_producer(stream_id: int, video_path: str, frame_queue: mp.Queue, stop_event: mp.Event, target_fps: int):
    """
    Reads video frames, resizes them, and pushes raw uint8 arrays to the queue.

    Optimization Note:
    We do NOT normalize or convert to Float32 here.
    Sending uint8 reduces Inter-Process Communication (IPC) overhead by 4x.
    """
    cap = SimulatedCamera(video_path, target_fps=target_fps)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        logger.error(f"[Stream {stream_id}] Failed to open video")
        return

    logger.info(f"[Stream {stream_id}] Started.")

    try:
        while not stop_event.is_set():
            ret, img = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # --- Preprocessing (CPU side) ---
            # Only Resize and Color conversion.
            # Keep as UINT8 to save bandwidth!
            img = cv2.resize(img, cfg.INPUT_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # --- Push to Queue ---
            try:
                # put_nowait raises queue.Full if full, allowing us to drop frames (Load Shedding)
                frame_queue.put_nowait((stream_id, img))
            except queue.Full:
                # Queue is full, meaning GPU is slower than Camera.
                # Drop this frame to maintain real-time latency.
                logger.warning(f"[Stream {stream_id}] Queue Full! Dropping frame.")
                pass

    except Exception as e:
        logger.error(f"[Stream {stream_id}] Error: {e}")
    finally:
        cap.release()
        logger.debug(f"[Stream {stream_id}] Stopped.")


# --- 2. Post-Processor ---
def result_consumer(service_id, result_queue, monitor_queue, stop_event):
    """Retrieves results, counts objects, overlays heatmap, and sends to Monitor."""
    logger.info(f"[Post-Process {service_id}] Started.")

    while not stop_event.is_set():
        try:
            # Get result from GPU
            data = result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        stream_id, density_map, original_img = data

        # A. Counting (Fast Integration)
        count = np.sum(density_map)

        # B. Visualization (Only if Monitor is enabled AND interval matches)
        # This saves CPU: we don't draw heatmaps for every single frame.
        if cfg.ENABLE_MONITOR:
            try:
                h, w = original_img.shape[:2]

                dm_norm = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                dm_norm = cv2.resize(dm_norm, (w, h))

                dm_color = (dm_norm * 255).astype(np.uint8)
                dm_color = cv2.applyColorMap(dm_color, cv2.COLORMAP_JET)

                # original_img is BGR here
                alpha = 0.5
                beta = 0.5

                threshold = 0.01

                mask = dm_norm > threshold
                overlay_img = original_img.copy()

                if mask.any():
                    roi_orig = original_img[mask]
                    roi_heat = dm_color[mask]

                    blended_roi = cv2.addWeighted(roi_orig, alpha, roi_heat, beta, 0)
                    overlay_img[mask] = blended_roi

                monitor_queue.put_nowait((stream_id, overlay_img, count))

            except queue.Full:
                pass  # Monitor is lagging, drop visualization frame
            except Exception as e:
                logger.error(f"Vis Error: {e}")


# --- 3. New Monitor Process ---
def monitor_service(monitor_queue, stop_event, num_streams, frame_size):
    """
    A dedicated process that renders a grid view of all streams.
    Using cv2.imshow here avoids blocking the inference pipeline.
    """
    logger.info("[Monitor] Dashboard started. Press 'q' to exit.")

    # Calculate layout
    rows, cols, tile_size = create_mosaic_layout(num_streams, frame_size)
    canvas_w = cols * tile_size[0]
    canvas_h = rows * tile_size[1]

    # Initialize a black canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Store the last known image for each stream to prevent flickering
    stream_cache = {}

    while not stop_event.is_set():
        # --- A. 接收数据阶段 ---
        updates_count = 0
        MAX_UPDATES_PER_FRAME = 100  # 【关键】：每次最多只取 100 个，防止死循环

        try:
            # 尽可能多地把队列里的数据取出来，更新缓存
            while True:
                # 接收三个数据：谁发的(id)，图是什么(img)，多少人(count)
                stream_id, img, count = monitor_queue.get_nowait()

                # 更新这一路的缓存，覆盖旧数据
                stream_cache[stream_id] = (img, count)

                updates_count += 1
                if updates_count >= MAX_UPDATES_PER_FRAME:
                    # 如果一次性处理太多了，为了不卡死 UI，强制跳出
                    break

        except queue.Empty:
            pass

        # --- B. 渲染阶段 (最关键的部分) ---
        # 遍历缓存中所有的流 (如果有22路，这里就会循环22次)
        for s_id, (img, count) in stream_cache.items():
            # 1. 根据 ID 算出这一路应该在大屏幕的哪个格子里
            # 比如 ID=0 -> 第0行，第0列
            # 比如 ID=7 -> 第1行，第1列
            r = s_id // cols  # 行号
            c = s_id % cols  # 列号

            # 2. 算出这个格子的【绝对像素坐标】 (x, y)
            x = c * tile_size[0]
            y = r * tile_size[1]

            # 3. 把这一路的小图片贴到对应的格子里
            # 先缩放
            img_tile = cv2.resize(img, tile_size)
            # 再粘贴
            canvas[y : y + tile_size[1], x : x + tile_size[0]] = img_tile

            # 4. 【画字】：这一步是分别画的！
            # 我们根据算出来的 (x, y)，把字画在各自格子的左上角
            label = f"ID:{s_id} | {int(count)}"

            # (可选) 画个黑色背景条，让字看清楚
            cv2.rectangle(canvas, (x, y), (x + 120, y + 30), (0, 0, 0), -1)

            # 写字：注意坐标是 (x+5, y+20)，这就保证了字永远跟着图走
            cv2.putText(
                canvas,
                label,
                (x + 5, y + 20),  # <--- 关键：每个 ID 的 x,y 都不一样
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,  # 字体大小统一
                (0, 255, 255),  # 黄色
                2,
            )

            # 画个绿框，区分边界
            cv2.rectangle(canvas, (x, y), (x + tile_size[0], y + tile_size[1]), (0, 255, 0), 2)

        # --- C. 显示阶段 ---
        cv2.imshow("Industrial Bird Crowd Monitor System", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()


def main_inference_loop():
    # Set multiprocessing start method to 'spawn' (Required for CUDA)
    mp.set_start_method("spawn", force=True)

    stop_event = mp.Event()

    # Shared Queues
    frame_queue = mp.Queue(maxsize=cfg.FRAME_QUEUE_SIZE)
    result_queue = mp.Queue(maxsize=cfg.RESULT_QUEUE_SIZE)
    monitor_queue = mp.Queue(maxsize=cfg.MONITOR_QUEUE_SIZE)

    # 1. Start Streams
    producers = []
    logger.info(f"Starting {cfg.NUM_STREAMS} video stream(s) at {cfg.TARGET_FPS} FPS each...")

    for i in range(cfg.NUM_STREAMS):
        p = mp.Process(target=frame_producer, args=(i, cfg.VIDEO_PATH, frame_queue, stop_event, cfg.TARGET_FPS))
        p.start()
        producers.append(p)

    # 2. Start Post-Processors
    # Usually 2-4 workers are enough for lightweight counting
    post_processors = []
    logger.info(f"Starting {cfg.NUM_POST_PROCESSES} post-processing service...")

    for i in range(cfg.NUM_POST_PROCESSES):
        p = mp.Process(target=result_consumer, args=(i, result_queue, monitor_queue, stop_event))
        p.start()
        post_processors.append(p)

    monitor_p = None
    if cfg.ENABLE_MONITOR:
        monitor_p = mp.Process(
            target=monitor_service, args=(monitor_queue, stop_event, cfg.NUM_STREAMS, cfg.INPUT_SIZE)
        )
        monitor_p.start()

    # 3. Initialize Model (GPU Side)
    device = torch.device(f"cuda:{cfg.DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading Model on {device}...")

    if cfg.USE_VGG:
        model = VGG19()
        ckpt = cfg.CKPT_VGG
    else:
        model = ShuffleNetV2_x1_0()
        ckpt = cfg.CKPT_SHUFFLE

    try:
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        logger.error(f"Checkpoint not found at {ckpt}. Please check path.")
        stop_event.set()
        return

    model.to(device).eval()

    # Enable CUDNN Benchmark for fixed size inputs (Significant Speedup)
    torch.backends.cudnn.benchmark = True

    # Pre-define normalization constants on GPU to avoid CPU processing
    # ImageNet Mean/Std
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    logger.info("Inference Engine Ready. Processing...")

    total_frames = 0
    last_log_time = time.time()

    try:
        while True:
            batch_arrays = []
            batch_meta = []  # Stores (stream_id, )

            # --- Step A: Batch Collection ---
            # Try to fill the batch. If queue is empty, proceed with what we have.

            while len(batch_arrays) < cfg.BATCH_SIZE:
                try:
                    # Adaptive timeout: wait less if we already have data
                    to = 0.002 if len(batch_arrays) > 0 else 0.05
                    item = frame_queue.get(timeout=to)

                    s_id, img_array = item
                    batch_arrays.append(img_array)
                    batch_meta.append((s_id,))
                except queue.Empty:
                    if len(batch_arrays) > 0:
                        break  # Process partial batch
                    else:
                        continue  # Keep waiting if completely empty

            if not batch_arrays:
                continue

            # --- Step B: GPU Transfer & Preprocessing (Industrial Optimization) ---

            # 1. Stack numpy arrays (CPU) -> Tensor (CPU)
            # Input shape: (B, H, W, 3) | uint8
            input_tensor_cpu = torch.from_numpy(np.stack(batch_arrays))

            # 2. Transfer to GPU (Asynchronous)
            # Permute to (B, 3, H, W)
            input_tensor = input_tensor_cpu.to(device, non_blocking=True).permute(0, 3, 1, 2)

            # 3. Cast to Float, Scale, and Normalize (All on GPU)
            # This is much faster than doing it in the DataLoader/Producer
            input_tensor = input_tensor.float().div(255.0)
            input_tensor = (input_tensor - mean) / std

            # --- Step C: Inference ---
            with torch.no_grad():
                # Automatic Mixed Precision (FP16)
                with torch.amp.autocast("cuda"):
                    outputs, _ = model(input_tensor)

            # --- Step D: Dispatch Results ---
            # Move only the result back to CPU
            outputs_np = outputs[:, 0].float().cpu().numpy()

            for i, density_map in enumerate(outputs_np):
                stream_id, *args = batch_meta[i]
                original_img = batch_arrays[i]

                try:
                    result_queue.put_nowait((stream_id, density_map, original_img, *args))
                except queue.Full:
                    # Post-processing is too slow; drop result to keep inference running
                    logger.warning(f"[Stream {stream_id}] Dropping result! Post-processing is too slow.")
                    pass

            batch_size_current = len(batch_arrays)
            total_frames += batch_size_current

            now = time.time()
            if now - last_log_time > 5.0:  # 每2秒打印一次
                fps = total_frames / (now - last_log_time)
                logger.info(f"[GPU Loop] Throughput: {fps:.1f} FPS (Target: {cfg.NUM_STREAMS * cfg.TARGET_FPS})")

                # 重置计数
                total_frames = 0
                last_log_time = now

    except KeyboardInterrupt:
        logger.info("Stopping Inference Engine...")
        stop_event.set()
    except Exception:
        logger.exception("Critical Error in Main Loop")
        stop_event.set()
    finally:
        # Cleanup
        for p in producers:
            p.terminate()
        for p in post_processors:
            p.terminate()
        if monitor_p:
            monitor_p.terminate()
        logger.info("System Shutdown Complete.")


if __name__ == "__main__":
    main_inference_loop()
