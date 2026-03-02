"""Pure luma/chroma fusion algorithm — no ROS dependencies.

Warps RGB to align with IR via ORB feature matching, then replaces
luma with IR while keeping chroma from the warped RGB. Runs the
color math on GPU via PyTorch.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_MIN_MATCHES = 10


class FusionPipeline:
    """IR luma + RGB chroma fusion with ORB warp alignment.

    Parameters
    ----------
    device : str
        PyTorch device string ("cuda" or "cpu").
    orb_features : int
        Number of ORB features to detect.
    ema_alpha : float
        EMA smoothing factor for the warp matrix.
    freeze_mode : str
        "auto" — EMA for *freeze_after* seconds, then freeze.
        "on"   — frozen immediately (identity if no cached M).
        "off"  — never freeze, EMA runs continuously.
    freeze_after : float
        Seconds of EMA before auto-freezing (only used when freeze_mode="auto").
    crop : bool
        If True, output is cropped to the valid RGB/IR overlap region.
        If False, output is the full IR frame with grayscale borders.
    """

    def __init__(
        self,
        device: str = "cuda",
        orb_features: int = 1000,
        ema_alpha: float = 0.2,
        freeze_mode: str = "auto",
        freeze_after: float = 5.0,
        crop: bool = False,
    ) -> None:
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._orb_features = orb_features
        self._ema_alpha = ema_alpha
        self.freeze_mode = freeze_mode
        self.freeze_after = freeze_after
        self.crop = crop

        # Warp state
        self._M: np.ndarray | None = None
        self._crop: tuple[int, int, int, int] | None = None
        self._grid: torch.Tensor | None = None
        self._grid_key: object = None  # (id(M), crop) cache key

        # Async warp worker
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._start_time = time.monotonic()
        self._frozen = False

        self._worker = threading.Thread(target=self._warp_worker, daemon=True)
        self._worker.start()

        # Per-frame timing breakdown (nanoseconds)
        self.timings: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, ir_gray: np.ndarray, rgb: np.ndarray, rgb_t: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Full pipeline: warp estimation -> GPU colorize -> RGB output.

        Parameters
        ----------
        ir_gray : (H, W) uint8 grayscale IR image.
        rgb : (H, W, 3) uint8 RGB color image.
        rgb_t : optional (1, 3, H, W) float32 GPU tensor of rgb.
            If provided, skips RGB upload (shared across pipelines).

        Returns
        -------
        (H', W', 3) uint8 RGB fused image (cropped to valid overlap region).
        Returns grayscale IR as RGB if no valid warp exists yet.
        """
        _t = time.perf_counter_ns
        H, W = ir_gray.shape[:2]

        # Submit warp estimation request (non-blocking)
        t0 = _t()
        self._submit_warp(rgb, ir_gray)
        t1 = _t()

        M = self._M
        if M is None:
            self.timings = {"submit": t1 - t0}
            return cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)

        crop = self._compute_crop(M, W, H)
        t2 = _t()
        if crop is None:
            self.timings = {"submit": t1 - t0, "crop": t2 - t1}
            return cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)

        # Full-frame mode: color where RGB covers, grayscale borders elsewhere
        roi = crop if self.crop else (0, 0, W, H)

        # GPU colorize — skip RGB upload if caller provided it
        if rgb_t is None:
            rgb_t = (
                torch.from_numpy(rgb)
                .to(self._device)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
        t3 = _t()
        result_t = self._gpu_colorize(rgb_t, ir_gray, M, roi, H, W)
        t4 = _t()
        out = result_t.clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        t5 = _t()

        self.timings = {
            "submit": t1 - t0,
            "crop": t2 - t1,
            "to_gpu": t3 - t2,
            "colorize": t4 - t3,
            "to_cpu": t5 - t4,
        }
        return out

    # ------------------------------------------------------------------
    # ORB warp estimation
    # ------------------------------------------------------------------

    def _estimate_warp(
        self,
        orb: cv2.ORB,
        bf: cv2.BFMatcher,
        rgb_gray: np.ndarray,
        ir_gray: np.ndarray,
    ) -> np.ndarray | None:
        kp1, des1 = orb.detectAndCompute(rgb_gray, None)
        kp2, des2 = orb.detectAndCompute(ir_gray, None)
        if (
            des1 is None
            or des2 is None
            or len(kp1) < _MIN_MATCHES
            or len(kp2) < _MIN_MATCHES
        ):
            return None
        matches = bf.match(des1, des2)
        if len(matches) < _MIN_MATCHES:
            return None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0,
        )
        return M

    # ------------------------------------------------------------------
    # Async warp worker
    # ------------------------------------------------------------------

    def _submit_warp(self, rgb: np.ndarray, ir_gray: np.ndarray) -> None:
        if self._frozen:
            return
        # Auto-freeze check
        if self.freeze_mode == "auto" and self._M is not None:
            if time.monotonic() - self._start_time > self.freeze_after:
                self._frozen = True
                return
        if self.freeze_mode == "on":
            self._frozen = True
            return
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        with self._lock:
            self._pending = (rgb_gray.copy(), ir_gray.copy())
        self._event.set()

    def _warp_worker(self) -> None:
        orb = cv2.ORB_create(nfeatures=self._orb_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        while True:
            self._event.wait()
            self._event.clear()
            with self._lock:
                pending = self._pending
                self._pending = None
            if pending is None:
                continue
            rgb_gray, ir_gray = pending
            M = self._estimate_warp(orb, bf, rgb_gray, ir_gray)
            if M is not None:
                if self._M is not None:
                    self._M = (1 - self._ema_alpha) * self._M + self._ema_alpha * M
                else:
                    self._M = M

    # ------------------------------------------------------------------
    # Crop computation
    # ------------------------------------------------------------------

    def _compute_crop(
        self, M: np.ndarray, W: int, H: int,
    ) -> tuple[int, int, int, int] | None:
        corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
        warped = cv2.transform(corners, M).reshape(4, 2)
        tl, tr, br, bl = warped
        x1 = int(np.ceil(max(tl[0], bl[0], 0)))
        y1 = int(np.ceil(max(tl[1], tr[1], 0)))
        x2 = int(np.floor(min(tr[0], br[0], W)))
        y2 = int(np.floor(min(bl[1], br[1], H)))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # GPU grid + colorize
    # ------------------------------------------------------------------

    def _build_grid(
        self, M: np.ndarray, crop: tuple[int, int, int, int], H: int, W: int,
    ) -> torch.Tensor:
        x1, y1, x2, y2 = crop
        M_full = np.vstack([M, [0, 0, 1]])
        M_inv = np.linalg.inv(M_full)[:2]

        cy, cx = torch.meshgrid(
            torch.arange(y1, y2, device=self._device, dtype=torch.float32),
            torch.arange(x1, x2, device=self._device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([cx, cy, torch.ones_like(cx)], dim=-1)
        M_inv_t = torch.tensor(M_inv, dtype=torch.float32, device=self._device)
        src = torch.einsum("ij,hwj->hwi", M_inv_t, coords)

        src[..., 0] = 2 * src[..., 0] / (W - 1) - 1
        src[..., 1] = 2 * src[..., 1] / (H - 1) - 1
        return src.unsqueeze(0)

    def _gpu_colorize(
        self,
        rgb_t: torch.Tensor,
        ir_gray: np.ndarray,
        M: np.ndarray,
        crop: tuple[int, int, int, int],
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Warp RGB via grid_sample, fuse luma from IR + chroma from RGB."""
        x1, y1, x2, y2 = crop

        # Cache grid — rebuild only when M or crop changes
        cache_key = (id(M), crop)
        if self._grid_key != cache_key:
            self._grid = self._build_grid(M, crop, H, W)
            self._grid_key = cache_key
        grid = self._grid

        ir_t = torch.from_numpy(ir_gray[y1:y2, x1:x2].copy()).to(self._device).float()

        warped = F.grid_sample(
            rgb_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        R, G, B = warped[0, 0], warped[0, 1], warped[0, 2]

        # Extract chroma from warped RGB
        Cr_off = 0.500 * R - 0.419 * G - 0.081 * B
        Cb_off = -0.169 * R - 0.331 * G + 0.500 * B

        # Fuse: IR luma + RGB chroma
        R_out = ir_t + 1.403 * Cr_off
        G_out = ir_t - 0.714 * Cr_off - 0.344 * Cb_off
        B_out = ir_t + 1.773 * Cb_off

        return torch.stack([R_out, G_out, B_out], dim=0)

    # ------------------------------------------------------------------
    # Runtime parameter updates
    # ------------------------------------------------------------------

    def set_freeze_mode(self, mode: str) -> None:
        """Update freeze mode at runtime. Resets freeze state if changed."""
        if mode == self.freeze_mode:
            return
        self.freeze_mode = mode
        if mode == "off":
            self._frozen = False
            self._event.set()
        elif mode == "on":
            self._frozen = True
        elif mode == "auto":
            self._frozen = False
            self._start_time = time.monotonic()

    def set_freeze_after(self, seconds: float) -> None:
        """Update auto-freeze duration at runtime."""
        self.freeze_after = seconds
        # If already frozen in auto mode, unfreeze to re-evaluate
        if self.freeze_mode == "auto" and self._frozen:
            self._frozen = False
            self._start_time = time.monotonic()
