import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import queue

import cv2
import numpy as np

from .config import Config
from .memory_manager import SharedMemoryInfo


logger = logging.getLogger(__name__)


class ResultProcessor(mp.Process):
    """
    Processes results from Workers.
    Optimized: Reads original frames from Shared Memory (Zero-Copy) instead of receiving them via Queue.
    Responsible for Unlocking Buffers after processing.
    """

    def __init__(
        self,
        result_queue,
        shutdown_event,
        handlers: list,
        shm_frames_info: SharedMemoryInfo,
        shm_meta_info: SharedMemoryInfo,
        cfg: Config,
    ):
        super().__init__(name="Result-Processor", daemon=True)
        self.result_queue = result_queue
        self.shutdown_event = shutdown_event
        self.handlers = handlers
        self.shm_frames_info = shm_frames_info
        self.shm_meta_info = shm_meta_info

        self.cfg = cfg

    def run(self):
        # 1. Attach Shared Memories
        try:
            existing_shm_frames = shm.SharedMemory(name=self.shm_frames_info.name)
            shm_frames = np.ndarray(
                self.shm_frames_info.shape, dtype=self.shm_frames_info.dtype, buffer=existing_shm_frames.buf
            )

            existing_shm_meta = shm.SharedMemory(name=self.shm_meta_info.name)
            shm_meta = np.ndarray(
                self.shm_meta_info.shape, dtype=self.shm_meta_info.dtype, buffer=existing_shm_meta.buf
            )

        except Exception as e:
            logger.error(f"[ResultProcessor] SHM Attach Failed: {e}")
            return

        for h in self.handlers:
            h.setup()

        logger.info("[Post-Processor] Started.")

        # Visualization Constants
        alpha, beta, threshold = 0.5, 0.5, 0.01

        while not self.shutdown_event.is_set():
            try:
                item = self.result_queue.get(timeout=0.005)
            except queue.Empty:
                continue

            try:
                # 2. Extract Data
                # item.outputs is the density map (B, 1, H, W) or similar
                # We need to perform visualization here to offload the GPU Worker

                final_vis_batch = []
                final_cnt_batch = []

                # Zero-Copy Read from Shared Memory
                # We use advanced indexing to get the batch of original frames
                batch_orig_frames = shm_frames[item.sids, item.buffer_indices]

                for i in range(len(item.sids)):
                    sid = item.sids[i]
                    buf_idx = item.buffer_indices[i]

                    # Visualization Logic
                    vis_map = item.outputs[i, 0]  # Assuming (B, 1, H, W) layout
                    orig_img = batch_orig_frames[i]

                    norm_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
                    norm_map = cv2.resize(norm_map, (orig_img.shape[1], orig_img.shape[0]))
                    mask = norm_map > threshold

                    overlay = orig_img.copy()
                    if mask.any():
                        dm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        dm_color = cv2.cvtColor(dm_color, cv2.COLOR_BGR2RGB)

                        blended = cv2.addWeighted(orig_img[mask], alpha, dm_color[mask], beta, 0)
                        overlay[mask] = blended

                    final_vis_batch.append(overlay)
                    final_cnt_batch.append(vis_map.sum())

                    # --- CRITICAL: UNLOCK BUFFER ---
                    # Mark buffer as Free (0.0) so Grabber can reuse it
                    shm_meta[sid, buf_idx, 0] = 0.0

                if final_vis_batch:
                    # Stack list to numpy array for the handler
                    vis_np = np.stack(final_vis_batch)
                    cnt_np = np.stack(final_cnt_batch)
                    for h in self.handlers:
                        h.handle(item.sids, vis_np, cnt_np, item.timestamp)

            except Exception as e:
                logger.error(f"Result Processor Error: {e}")
                import traceback

                traceback.print_exc()

        for h in self.handlers:
            h.cleanup()
