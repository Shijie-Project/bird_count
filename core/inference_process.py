import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.shufflenet import get_shufflenet_density_model

from .config import Config
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import get_optimal_memory_format, setup_logging


# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """
    Data Transfer Object (DTO) passed from Processor to Consumer.
    """

    stream_id: int
    buffer_idx: int
    frame_idx: int
    timestamp: float
    count: float
    latency: float


@dataclass
class BatchInferenceResult:
    results: list[InferenceResult]

    def append(self, result: InferenceResult):
        self.results.append(result)


class InferenceProcess(mp.Process):
    """
    The Processing Unit.
    Reads READY frames from Shared Memory, batches them, performs inference,
    and pushes results to the Output Queue.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, result_queue: mp.Queue):
        super().__init__(name="InferenceEngine")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self._stop_event = mp.Event()

        self.device = None
        self.model = None
        self.mean = None
        self.std = None
        self._memory_format = None
        self._colormap_lut = None  # Lookup table for GPU ColorMap

        self._interval = config.frame_interval

    def _init_resource(self):
        try:
            # A. GPU Setup
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

            # B. Model Loading & Warmup
            self.device = torch.device(self.config.device)
            # Normalization constants
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            self._memory_format = get_optimal_memory_format(self.device)

            # C. Initialize ColorMap Lookup Table (LUT) on GPU
            self.colormap_lut = self._create_colormap_lut().to(self.device)

            self._load_and_warmup_model()

            # D. Connect to Shared Memory
            self.shm_client = SharedMemoryClient(self.shm_config)
            self.shm_client.connect()

            logger.info("Ready to process frames.")

        except Exception as e:
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            return

    def _create_colormap_lut(self) -> torch.Tensor:
        """
        Creates a (256, 3) lookup table for JET colormap to be used on GPU.
        Returns:
            Tensor [256, 3] uint8
        """
        # Create a gradient 0-255
        gradient = np.arange(256, dtype=np.uint8).reshape(1, 256)
        # Use OpenCV to generate the colors (BGR)
        colormap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        # Remove extra dim -> (256, 3)
        colormap = colormap.squeeze(0)
        return torch.tensor(colormap, dtype=torch.uint8)

    def _load_and_warmup_model(self):
        logger.info(f"Loading Model on {self.device}...")
        self.model = get_shufflenet_density_model(
            model_path=self.config.model.path, device=self.device, inference=True
        )
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        except Exception:
            logger.warning("Model compilation failed. Falling back to eager mode.")

        self.model.eval()

        # logger.info("Starting GPU Warmup...")
        # with torch.inference_mode():
        #     for bsz in range(1, self.config.num_streams + 1):
        #         # Create dummy frames [B, H, W, 3]
        #         dummy_frames = np.zeros((bsz, *self.shm_config.shape[2:]), dtype=np.uint8)
        #         # Run full pipeline including overlay
        #         input_tensor, raw_tensor = self._preprocess_batch_on_gpu(dummy_frames)
        #         with torch.amp.autocast("cuda"):
        #             _ = self.model(input_tensor)
        #         logger.debug(f"Warmup Batch Size {bsz} Complete.")
        #
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # logger.info("Warmup Complete.")

    def _preprocess_batch_on_gpu(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert List[np.ndarray] to GPU Tensors.
        Returns:
            input_tensor: [B, 3, H, W] float normalized (for model)
            raw_tensor:   [B, H, W, 3] uint8 (for visualization)
        """
        # 1. Zero-Copy to Tensor (Stack creates a copy)
        # [B, H, W, 3]
        raw_tensor = torch.from_numpy(frames).to(self.device, non_blocking=True)

        # 2. Prepare Input for Model (NHWC -> NCHW, Float, Normalize)
        input_tensor = raw_tensor.permute(0, 3, 1, 2).float().div(255.0)
        input_tensor = (input_tensor - self.mean) / self.std

        # Return both
        return input_tensor, raw_tensor

    def _apply_overlays_on_gpu(
        self, density_map: torch.Tensor, raw_images: torch.Tensor, alpha: float = 0.6, beta: float = 0.4
    ) -> torch.Tensor:
        """
        Batch processing of Heatmap Overlay on GPU.
        Args:
            density_map: [B, 1, h, w] Output from model
            raw_images:  [B, H, W, 3] Original images on GPU (uint8)
        Returns:
            blended_cpu: [B, H, W, 3] uint8 numpy array
        """
        B, H, W, C = raw_images.shape

        # 1. Upsample Density Map (Bilinear)
        # [B, 1, h, w] -> [B, 1, H, W]
        heatmaps = F.interpolate(density_map, size=(H, W), mode="bilinear", align_corners=False)

        # 2. Normalize to 0-1 (Per image min/max)
        # Flatten spatial dims: [B, H*W]
        v_min = heatmaps.amin(dim=(1, 2, 3), keepdim=True)
        v_max = heatmaps.amax(dim=(1, 2, 3), keepdim=True)
        norm_maps = (heatmaps - v_min) / (v_max - v_min + 1e-5)

        indices = (norm_maps.squeeze(1) * 255.9).long()  # [B, H, W]
        heatmap_rgb = self.colormap_lut[indices]  # [B, H, W, 3]

        mask = norm_maps.permute(0, 2, 3, 1) > 0.01

        out = raw_images.clone().float()

        out = torch.where(mask, out * alpha + heatmap_rgb.float() * beta, out)

        return out.clamp(0, 255).to(torch.uint8)

    def _collect_batch_from_shm(self):
        stream_ids = []
        buffer_indices = []

        # Round-robin collection
        for stream_id in range(self.config.num_streams):
            b_idx = self._get_latest_frame(stream_id)

            if b_idx != -1:
                self.shm_client.metadata[stream_id, b_idx]["state"] = BufferState.READING

                stream_ids.append(stream_id)
                buffer_indices.append(b_idx)

        batch_frames = self.shm_client.frames[stream_ids, buffer_indices]
        batch_meta = self.shm_client.metadata[stream_ids, buffer_indices]

        return batch_frames, batch_meta, stream_ids, buffer_indices

    def _get_latest_frame(self, stream_id):
        # ... (Previous Logic: Best-Effort Scan) ...
        best_idx = -1
        max_frame_idx = -1
        dropped_info = []

        for i in range(self.config.num_buffers):
            meta = self.shm_client.metadata[stream_id, i]
            if meta["state"] == BufferState.READY:
                current_f_idx = meta["frame_idx"]
                if current_f_idx > max_frame_idx:
                    if best_idx != -1:
                        # Drop stale candidate
                        prev_meta = self.shm_client.metadata[stream_id, best_idx]
                        dropped_info.append(f"#{prev_meta['frame_idx']}")
                        prev_meta["state"] = BufferState.FREE
                    max_frame_idx = current_f_idx
                    best_idx = i
                else:
                    # Drop stale
                    dropped_info.append(f"#{current_f_idx}")
                    meta["state"] = BufferState.FREE

        if dropped_info:
            # logger.debug(f"[Stream {stream_id}] Lag. Selected #{max_frame_idx}. Dropped: {dropped_info}")
            pass
        return best_idx

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info("Process starting...")
        self._init_resource()

        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()

                # --- 1. Batch Assembly ---
                # Added sids/b_idxs for potential batch operations
                batch_frames, batch_meta, stream_ids, buffer_indices = self._collect_batch_from_shm()

                if batch_frames.shape[0] == 0:
                    time.sleep(0.001)
                    continue

                # --- 2. Preprocess (GPU) ---
                # Now returns raw_tensor too (BHWC) for visualization
                input_tensor, raw_tensor = self._preprocess_batch_on_gpu(batch_frames)
                t1 = time.perf_counter()

                # --- 3. Inference & Visualization (GPU) ---
                with torch.inference_mode():
                    # A. Inference
                    with torch.amp.autocast("cuda"):
                        density_map = self.model(input_tensor)

                    # B. Counting
                    counts_tensor = torch.sum(density_map, dim=(1, 2, 3))
                    counts_cpu = counts_tensor.cpu().numpy().tolist()

                    # C. Batch Visualization (Overlay)
                    # This happens entirely on GPU and returns blended images (CPU numpy)
                    if self.config.envs.show_density_map:
                        blended_frames = self._apply_overlays_on_gpu(density_map, raw_tensor)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                t2 = time.perf_counter()
                process_time = time.time()

                # --- 4. Write Back Overlays (In-Place) ---
                # Write blended images back to Shared Memory
                if self.config.envs.show_density_map:
                    blended_numpy = blended_frames.cpu().numpy()
                    self.shm_client.frames[stream_ids, buffer_indices] = blended_numpy

                # --- 5. Package Results ---
                current_batch_results = BatchInferenceResult(results=[])

                for i, counts in enumerate(counts_cpu):
                    meta = batch_meta[i]
                    # Create result object
                    res = InferenceResult(
                        stream_id=int(meta["stream_id"]),
                        buffer_idx=int(meta["buffer_idx"]),
                        frame_idx=int(meta["frame_idx"]),
                        timestamp=float(meta["timestamp"]),
                        count=counts,
                        latency=process_time - float(meta["timestamp"]),
                    )
                    current_batch_results.append(res)

                # --- 6. Dispatch ---
                if current_batch_results.results:
                    try:
                        self.result_queue.put(current_batch_results, timeout=0.01)
                    except queue.Full:
                        logger.warning(f"Queue Full! Dropping batch of {len(current_batch_results.results)} frames.")
                        self.shm_client.metadata[stream_ids, buffer_indices]["state"] = BufferState.FREE

                self._log_performance(batch_meta, t0, t1, t2)

                # --- FPS Control ---
                elapsed = time.perf_counter() - t0
                wait_time = self._interval - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Loop Error: {e}", exc_info=True)
                time.sleep(0.1)

        self.shm_client.close()
        logger.info("Engine Stopped.")

    def stop(self):
        self._stop_event.set()

    def _log_performance(self, batch_meta: np.ndarray, t0: float, t1: float, t2: float):
        """
        Logs detailed timing metrics for the inference pipeline.

        Args:
            batch_meta: List of tuples (stream_id, buffer_idx, timestamp)
            t0: Start time (before preprocessing)
            t1: Time after GPU transfer/preprocessing
            t2: Time after model inference
        """
        # t3: End of this logging cycle (Post-processing start)
        t3 = time.perf_counter()

        # Wall clock time for latency calculation
        wall_now = time.time()

        batch_size = len(batch_meta)
        if batch_size == 0:
            return

        # 1. Extract Metadata
        sids = [int(m["stream_id"]) for m in batch_meta]
        timestamps = [m["timestamp"] for m in batch_meta]

        # 2. Calculate Latency (Frame Age)
        # How old is the data? (From Capture time to Now)
        # Using average age of the batch
        avg_age_ms = sum([(wall_now - ts) for ts in timestamps]) / batch_size * 1000

        # 3. Calculate Durations
        # Data Prep: CPU -> GPU Transfer + Resize/Normalize
        prep_ms = (t1 - t0) * 1000
        # Inference: CUDA Compute time
        infer_ms = (t2 - t1) * 1000
        # Post-proc / Total Loop time
        total_ms = (t3 - t0) * 1000

        # 4. Format Log Message
        log_msg = (
            f"Batch={batch_size} | "
            f"SIDs={sids} | "
            f"Age={avg_age_ms:.1f}ms | "  # latency
            f"Prep={prep_ms:.1f}ms | "  # pcie/mem
            f"Infer={infer_ms:.1f}ms | "  # gpu compute
            f"Total={total_ms:.1f}ms | "  # throughput
        )

        # Use debug level to avoid spamming production logs
        logger.debug(log_msg)
