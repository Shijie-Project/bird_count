import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
import warnings
from dataclasses import dataclass
from pprint import pformat
from typing import Optional

import cv2
import numpy as np
import torch

from models import ShuffleNetV2_x1_0
from utils import SimulatedCamera


logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "0") == "0" else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InferenceEngine")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# --- Define Types ---
GrabberItem = tuple[int, np.ndarray, float]
TaskItem = tuple[list[int], torch.Tensor]
ResultItem = tuple[list[int], torch.Tensor, torch.Tensor]


# --- Setup Logging ---
@dataclass
class Config:
    # Model
    model_path: str = "./ckpts/shufflenet_model_best.pth"

    # GPUs
    available_gpus: tuple[int, ...] = (0,)
    workers_per_gpu: int = 1  # usually 1 per GPU

    # Streams
    use_video: bool = False
    sources: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)
    video_sources: tuple[str, ...] = ("./data/bird_count/bird_count_demo.mp4",)

    num_streams: int = 22
    target_fps_per_stream: float = 10.0  # load-test knob (per stream)

    # Preprocess
    input_size: tuple[int, int] = (512, 512)

    # Video Capture
    open_timeout_ms: int = 5000  # best-effort: some backends ignore this
    runtime_seconds: Optional[int] = None  # None -> run until Ctrl+C

    # Monitoring
    enable_monitor: bool = True

    def __post_init__(self):
        stream_sources = self.video_sources if self.use_video else self.sources
        if len(stream_sources) == 0:
            raise ValueError("No stream sources found.")

        if len(stream_sources) < self.num_streams:
            repeats = self.num_streams // len(stream_sources) + 1
            stream_sources = tuple(stream_sources) * repeats
        self.stream_sources = stream_sources[: self.num_streams]

        assert len(self.stream_sources) == self.num_streams
        logging.info(f"Stream sources:\n{pformat(self.stream_sources)}")


# --- Frame Grabber (Thread) ---
class FrameGrabber(threading.Thread):
    def __init__(
        self,
        stream_id: int,
        src: str,
        input_size: tuple[int, int],
        frame_queue: "mp.Queue[GrabberItem]",
        stop_event: threading.Event,
        target_fps: float,
        open_timeout_ms: int,
    ) -> None:
        super().__init__(daemon=True, name=f"grabber-{stream_id}")
        self.stream_id = stream_id
        self.src = src
        self.input_size = input_size
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.target_fps = target_fps
        self.open_timeout_ms = open_timeout_ms

        self._target_period_s = 1.0 / max(1.0, target_fps)
        self._start_time = time.perf_counter()
        self._last_emit = time.perf_counter()

    def run(self) -> None:
        while not self.stop_event.is_set():
            # --- 1. Reconnection Loop ---
            if os.path.exists(self.src):
                cap = SimulatedCamera(self.src, target_fps=self.target_fps)
            else:
                cap = cv2.VideoCapture(self.src)

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.open_timeout_ms)
            except Exception:
                pass

            if not cap.isOpened():
                logging.error(f"Stream {self.stream_id}: Failed to open {self.src}. Retrying in 5s...")
                cap.release()
                time.sleep(5)
                continue

            logging.info(f"Stream {self.stream_id}: Connected.")

            # --- 2. Reading Loop ---
            last_success_time = time.perf_counter()

            while not self.stop_event.is_set():
                ret = cap.grab()

                if not ret:
                    if time.perf_counter() - last_success_time > 5:
                        # if do not receive any frames for the last 5 seconds, assume connection lost and reconnect
                        logging.warning(f"Stream {self.stream_id}: Connection lost. Reconnecting...")
                        break  # reconnect

                    time.sleep(0.01)
                    continue  # re-grab

                now = time.perf_counter()
                last_success_time = now

                if now - self._last_emit < self._target_period_s:
                    time.sleep(0.001)
                    continue  # re-grap

                ret, frame = cap.retrieve()
                if not ret:
                    continue

                frame = cv2.resize(frame, self.input_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    self.frame_queue.put_nowait((self.stream_id, frame, time.time()))
                    self._last_emit = now

                except queue.Full:
                    if now - self._start_time > 10.0:
                        logger.debug(f"Stream {self.stream_id}: Queue full, dropped old frame.")

                except Exception as e:
                    logging.error(f"Stream {self.stream_id} processing error: {e}")

            cap.release()

        logging.info(f"Stream {self.stream_id}: Disconnected!")


# --- Inference worker (process) ---
def inference_worker(
    worker_id: int,
    device_index: int,
    cfg: Config,
    task_queue: "mp.Queue[TaskItem]",
    result_queue: "mp.Queue[ResultItem]",
    shutdown: "mp.Event",
) -> None:
    torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{device_index}")
    logging.info(f"Worker {worker_id}: connected on {device}.")

    model = ShuffleNetV2_x1_0().to(device)
    state = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    while not shutdown.is_set():
        try:
            batch_data = task_queue.get(timeout=0.1)
        except Exception:
            continue

        stream_ids, x_cpu = batch_data
        x_cpu = x_cpu.float().permute(0, 3, 1, 2)

        input_tensor = x_cpu.to(device, non_blocking=True).div(255.0)
        input_tensor = (input_tensor - mean) / std

        t_infer_start = time.perf_counter()
        try:
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    outputs, _ = model(input_tensor)
                # Ensure synchronization if timing is critical
                torch.cuda.synchronize()

            t_infer_done = time.perf_counter()
            logger.debug(f"processed with {t_infer_done - t_infer_start:.3f}s")

            out_cpu = outputs.detach().float().cpu()

            # Send results back
            try:
                result_queue.put_nowait((stream_ids, x_cpu, out_cpu))
            except queue.Full:
                logger.debug("Result queue full, dropping batch.")
            except Exception as e:
                logger.error(f"Worker {worker_id}: Failed to put result. Error: {e}")

        except Exception as e:
            logging.error(f"Inference error: {e}")

    logging.info(f"Worker {worker_id} shutting down")


def run(cfg: Config) -> None:
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()

    # Stream grabbers -> per stream latest queue (size=1)
    per_stream_queues: list["mp.Queue[GrabberItem]"] = [ctx.Queue(maxsize=1) for _ in range(cfg.num_streams)]

    stop_event = threading.Event()
    grabbers: list[FrameGrabber] = []
    for sid, (src, q) in enumerate(zip(cfg.stream_sources, per_stream_queues)):
        t = FrameGrabber(
            stream_id=sid,
            src=src,
            frame_queue=q,
            input_size=cfg.input_size,
            stop_event=stop_event,
            target_fps=cfg.target_fps_per_stream,
            open_timeout_ms=cfg.open_timeout_ms,
        )
        t.start()
        grabbers.append(t)

    # GPU workers
    gpu_worker_specs: list[tuple[int, int]] = []
    for gpu in cfg.available_gpus:
        for _ in range(max(1, cfg.workers_per_gpu)):
            gpu_worker_specs.append((gpu, len(gpu_worker_specs)))

    result_queue: "mp.Queue[ResultItem]" = ctx.Queue(maxsize=1)
    task_queues: list["mp.Queue[TaskItem]"] = [ctx.Queue(maxsize=1) for _ in gpu_worker_specs]

    workers: list[mp.Process] = []
    for (gpu, wid), tq in zip(gpu_worker_specs, task_queues):
        p = ctx.Process(
            target=inference_worker,
            args=(wid, gpu, cfg, tq, result_queue, shutdown),
            name=f"gpu-worker-{wid}-cuda{gpu}",
            daemon=True,
        )
        p.start()
        workers.append(p)

    time.sleep(5)  # wait for setup

    # SIG handler
    def _handle_stop(_sig: int, _frame: object) -> None:
        stop_event.set()
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # Stream -> worker mapping (round-robin)
    stream_to_worker = {sid: sid % len(task_queues) for sid in range(cfg.num_streams)}

    t_start_wall = time.time()
    try:
        while not shutdown.is_set():
            if cfg.runtime_seconds is not None and (time.time() - t_start_wall) >= cfg.runtime_seconds:
                logging.info("Reached runtime_seconds=%s, stopping.", cfg.runtime_seconds)
                break

            worker_batches: list[list[GrabberItem]] = [[] for _ in range(len(task_queues))]

            # Check all stream queues
            for sid, q in enumerate(per_stream_queues):
                try:
                    # Get frame if available
                    item = q.get_nowait()  # (sid, tensor)
                    widx = stream_to_worker[sid]
                    worker_batches[widx].append(item)
                except Exception:
                    continue

            # Send Batches to Workers
            did_work = False
            for widx, batch_items in enumerate(worker_batches):
                if not batch_items:
                    continue

                did_work = True
                received_frames = sum(len(batch) > 0 for batch in batch_items)
                logger.debug(f"Worker{widx}: Received {received_frames} frames.")

                # Construct the batch properly here
                sids = [x[0] for x in batch_items]
                tensors = torch.from_numpy(np.stack([x[1] for x in batch_items]))

                try:
                    task_queues[widx].put_nowait((sids, tensors))
                except Exception:
                    logger.debug("Task queue full, dropping batch.")

            while True:
                try:
                    _ = result_queue.get_nowait()
                except Exception:
                    break

            if not did_work:
                time.sleep(0.001)  # Prevent CPU spin

    finally:
        stop_event.set()
        shutdown.set()

        # teardown workers
        for _ in range(60):
            if all(not p.is_alive() for p in workers):
                break
            time.sleep(0.05)

        for p in workers:
            if p.is_alive():
                p.terminate()
            p.join(timeout=1.0)

        logging.info("Stopped.")


def main() -> None:
    # You can set CUDA_VISIBLE_DEVICES="0,1" etc.
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip()
    gpus = tuple(int(x) for x in visible.split(",") if x.strip() != "")
    cfg = Config(
        available_gpus=gpus if gpus else (0,),
        workers_per_gpu=1,
        num_streams=4,
        target_fps_per_stream=10.0,
        use_video=os.getenv("USE_VIDEO", "0") == "1",
        runtime_seconds=None,
        enable_monitor=True,
    )
    run(cfg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
