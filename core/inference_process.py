import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from models import get_model

from .config import Config
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import create_colormap_lut, get_optimal_memory_format, setup_cuda, setup_logging


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """DTO for a single frame result. Note: frozen=False for flexible post-processing."""

    stream_id: int
    buffer_idx: int
    frame_idx: int
    timestamp: float
    count: float
    latency: float
    alert_flag: bool = False


@dataclass(slots=True)
class BatchInferenceResult:
    """Container for batch results. Optimized to reduce instantiation overhead."""

    results: list[InferenceResult] = field(default_factory=list)

    def append(self, result: InferenceResult):
        self.results.append(result)

    @property
    def stream_ids(self) -> list[int]:
        """Returns a list of stream IDs in the batch."""
        return [res.stream_id for res in self.results]

    @property
    def buffer_indices(self) -> list[int]:
        """Returns a list of SHM buffer indices in the batch."""
        return [res.buffer_idx for res in self.results]

    def __len__(self) -> int:
        return len(self.results)


class InferenceProcess(mp.Process):
    """
    High-Performance Inference Engine.
    Processes assigned video streams using GPU acceleration and Shared Memory.
    """

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        result_queue: mp.Queue,
        worker_id: int,
        total_workers: int,
    ):
        # Unique name for tracing
        super().__init__(name=f"InferenceWorker-{worker_id}")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.total_workers = total_workers
        self._stop_event = mp.Event()

        # --- Stream Partitioning ---
        # Logic: stream_id % total_workers == worker_id
        self.assigned_streams = [s for s in range(self.config.num_streams) if s % total_workers == worker_id]

        # Placeholders for resources initialized in run()
        self.device = None
        self.model = None
        self.shm_client = None

        # Optimization Constants
        self._fused_scale = None
        self._fused_bias = None
        self._memory_format = None
        self._colormap_lut = None
        self._interval = config.frame_interval

    def _init_resource(self):
        """Initializes GPU and SHM resources within the child process context."""
        try:
            setup_cuda()
            self.device = torch.device(self.config.device)

            # Detect optimal memory layout (NHWC for Tensor Cores)
            self._memory_format = get_optimal_memory_format(self.device)

            # Pre-calculate Fused Normalization: x * scale + bias
            # This replaces (x/255 - mean) / std with a single MAD operation
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            self._fused_scale = 1.0 / (255.0 * std)
            self._fused_bias = -(mean / std)

            # Load Model with Structural Fusion
            self.model = get_model(
                self.config.model.type, model_path=self.config.model.path, device=self.device, fuse=True
            )
            self.model.eval()
            # try:
            #     # 'reduce-overhead' is ideal for small models like ShuffleNet
            #     self.model = torch.compile(self.model, mode="reduce-overhead")
            #     logger.info(f"[{self.name}] Torch compilation enabled.")
            # except Exception as e:
            #     logger.warning(f"[{self.name}] Torch.compile failed: {e}. Using Eager mode.")

            # Connect to Shared Memory
            self.shm_client = SharedMemoryClient(self.shm_config)
            self.shm_client.connect()

            self._colormap_lut = create_colormap_lut().to(self.device)
            self._warmup_gpu()

            logger.info(f"[{self.name}] Initialization complete. Ready for streams {self.assigned_streams}")

        except Exception as e:
            logger.critical(f"[{self.name}] Startup failed: {e}", exc_info=True)
            raise

    def _warmup_gpu(self):
        """Warmup largest expected batch sizes to prevent startup latency spikes."""
        logger.info(f"[{self.name}] Starting GPU Warmup...")
        max_bsz = len(self.assigned_streams)
        warmup_sizes = sorted({max(1, max_bsz - i) for i in range(3)})

        with torch.inference_mode():
            for bsz in warmup_sizes:
                dummy = np.zeros((bsz, *self.shm_config.shape[2:]), dtype=np.uint8)
                input_t, _ = self._preprocess_batch_on_gpu(dummy)
                with torch.amp.autocast("cuda"):
                    _ = self.model(input_t)

                logger.debug(f"[{self.name}] Warmup for Batch Size {bsz} complete.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _preprocess_batch_on_gpu(self, frames: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert NumPy batch to GPU Tensors with fused normalization."""
        # Non-blocking transfer to GPU (H2D)
        raw_tensor = torch.from_numpy(frames).to(self.device, non_blocking=True)

        rgb_tensor = torch.flip(raw_tensor, dims=[-1])

        # Convert layout to optimal format while keeping logical NCHW for the model
        # .float() + In-place MAD (Multiply-Add)
        input_tensor = rgb_tensor.permute(0, 3, 1, 2).to(memory_format=self._memory_format).float()
        input_tensor.mul_(self._fused_scale).add_(self._fused_bias)

        return input_tensor, rgb_tensor

    def _apply_overlays_on_gpu(
        self, density_map: torch.Tensor, raw_images: torch.Tensor, alpha: float = 0.6, beta: float = 0.4
    ) -> torch.Tensor:
        """High-performance Heatmap Overlay using Half-Precision and Boolean Indexing."""
        B, H, W, C = raw_images.shape

        min_max = torch.aminmax(density_map.flatten(2), dim=2, keepdim=True)
        v_min, v_max = min_max.min.unsqueeze(-1), min_max.max.unsqueeze(-1)  # [B, 1, 1, 1]

        norm_maps_low = (density_map - v_min) / (v_max - v_min + 1e-5)

        # 1. Faster Upsampling
        norm_maps = F.interpolate(norm_maps_low, size=(H, W), mode="bilinear", align_corners=False)

        # 3. LUT Color Mapping
        indices = (norm_maps.squeeze(1) * 255.9).long()
        heatmap_rgb = self._colormap_lut[indices]  # [B, H, W, 3]

        # 4. In-place Blending using Half-Precision to save bandwidth
        mask = norm_maps.squeeze(1) > 0.01
        res = raw_images.to(torch.float16)  # Result buffer

        if mask.any():
            # Boolean indexing limits computation to significant pixels
            h_rgb_half = heatmap_rgb[mask].half()
            res[mask] = res[mask].mul_(alpha).add_(h_rgb_half, alpha=beta)

        return res.to(torch.uint8)

    def _collect_batch_from_shm(self) -> tuple[np.ndarray, np.ndarray, list, list]:
        """Vectorized collection of ready frames from SHM."""
        ready_pairs = [
            (s_id, b_idx) for s_id in self.assigned_streams if (b_idx := self._get_latest_frame(s_id)) != -1
        ]

        if not ready_pairs:
            return np.array([]), np.array([]), [], []

        sids, b_idxs = map(list, zip(*ready_pairs))

        # Bulk update metadata state to READING (minimizes memory barriers)
        self.shm_client.metadata[sids, b_idxs]["state"] = BufferState.READING

        # Vectorized fetching of frames and metadata
        batch_frames = self.shm_client.frames[sids, b_idxs]
        batch_meta = self.shm_client.metadata[sids, b_idxs]

        return batch_frames, batch_meta, sids, b_idxs

    def _get_latest_frame(self, stream_id: int) -> int:
        """Selects the newest READY frame and clears stale buffers in bulk."""
        stream_meta = self.shm_client.stream_metadata[stream_id]
        ready_mask = stream_meta["state"] == BufferState.READY

        if not ready_mask.any():
            return -1

        ready_indices = np.where(ready_mask)[0]
        ready_f_indices = stream_meta["frame_idx"][ready_mask]

        # Pick max frame_idx
        best_local_idx = np.argmax(ready_f_indices)
        best_idx = ready_indices[best_local_idx]

        # Bulk free stale frames
        stale_mask = ready_mask.copy()
        stale_mask[best_idx] = False
        if stale_mask.any():
            stream_meta["state"][stale_mask] = BufferState.FREE

        return best_idx

    def run(self):
        """Main Inference Loop."""
        setup_logging(self.config.envs.debug)
        self._init_resource()

        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()

                # --- 1. Collection ---
                batch_frames, batch_meta, sids, b_idxs = self._collect_batch_from_shm()
                if batch_frames.size == 0:
                    time.sleep(0.001)
                    continue

                # --- 2. GPU Preprocessing ---
                input_tensor, raw_tensor = self._preprocess_batch_on_gpu(batch_frames)
                t1 = time.perf_counter()

                # --- 3. Inference & Rendering ---
                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    # Model Inference
                    density_map = self.model(input_tensor)
                    # Parallel sum for counts
                    counts_tensor = torch.sum(density_map, dim=(1, 2, 3))

                    # Conditional visualization
                    if self.config.envs.show_density_map:
                        blended_gpu = self._apply_overlays_on_gpu(density_map, raw_tensor)

                # --- 4. Data Transfer (D2H) ---
                # Move counts to CPU (Tiny transfer, fast)
                counts_cpu = counts_tensor.cpu().numpy()

                # Move visualization frames to SHM (Large transfer, slow)
                if self.config.envs.show_density_map:
                    # Async copy back to SHM slice
                    self.shm_client.frames[sids, b_idxs] = blended_gpu.cpu().numpy()  # RGB

                t2 = time.perf_counter()
                process_time = time.time()

                # --- 5. Packaging (Optimized List Comprehension) ---
                batch_packet = BatchInferenceResult(
                    results=[
                        InferenceResult(
                            stream_id=int(batch_meta[i]["stream_id"]),
                            buffer_idx=int(batch_meta[i]["buffer_idx"]),
                            frame_idx=int(batch_meta[i]["frame_idx"]),
                            timestamp=float(batch_meta[i]["timestamp"]),
                            count=float(counts_cpu[i]),
                            latency=process_time - float(batch_meta[i]["timestamp"]),
                        )
                        for i in range(len(sids))
                    ]
                )

                # --- 6. Dispatch ---
                try:
                    self.result_queue.put(batch_packet, timeout=0.01)
                except queue.Full:
                    # Critical: Release SHM buffers if consumer is blocked to prevent deadlock
                    self.shm_client.metadata[sids, b_idxs]["state"] = BufferState.FREE
                    logger.warning(f"[{self.name}] Output Queue Full. Dropping {len(sids)} frames.")

                self._log_performance(batch_meta, t0, t1, t2)

                # FPS Control
                elapsed = time.perf_counter() - t0
                if elapsed < self._interval:
                    time.sleep(self._interval - elapsed)

            except Exception as e:
                logger.error(f"[{self.name}] Loop Error: {e}", exc_info=True)
                time.sleep(0.01)

        self.shm_client.disconnect()
        logger.info(f"[{self.name}] Stopped.")

    def stop(self):
        self._stop_event.set()

    def _log_performance(self, batch_meta: np.ndarray, t0: float, t1: float, t2: float):
        """Calculates and logs throughput metrics."""
        t3 = time.perf_counter()
        now = time.time()
        batch_size = len(batch_meta)

        avg_age = sum([(now - m["timestamp"]) for m in batch_meta]) / batch_size * 1000
        prep_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        total_ms = (t3 - t0) * 1000

        logger.debug(
            f"[{self.name}] Batch={batch_size} | Age={avg_age:.1f}ms | "
            f"Prep={prep_ms:.1f}ms | Infer={infer_ms:.1f}ms | Total={total_ms:.1f}ms"
        )
