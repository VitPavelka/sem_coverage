# sem_coverage.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import logging
import numpy as np

from tifffile import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, profile_line
from skimage.morphology import (
    closing,
    disk,
    erosion,
    opening,
    white_tophat,
)

# NOTE: remove_small_holes/remove_small_objects signatures changed in skimage>=0.26
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries


# -----------------------------
# Exceptions
# -----------------------------
class SEMAnalysisError(Exception):
    """Base class for SEM analysis errors."""


class ImageLoadError(SEMAnalysisError):
    """Raised when an input image cannot be loaded or has an unsupported format."""


class PreprocessError(SEMAnalysisError):
    """Raised when preprocessing (crop/normalize) fails."""


class SegmentationError(SEMAnalysisError):
    """Raised when segmentation fails."""


# -----------------------------
# Config + Result containers
# -----------------------------
@dataclass(frozen=True)
class AnalyzerConfig:
    """
    Configuration parameters for SEM bead coverage analysis.

    Notes
    -----
    - Coverage is measured in the 2D projection: Ag pixels / bead pixels.
    - Ag segmentation uses white top-hat to suppress slow background variations
      (illumination/shading across the bead) and enhance small bright features.
    """

    # Info-bar cropping
    infobar_tail_rows: int = 320
    infobar_k_mad: float = 8.0
    infobar_min_run: int = 10

    # Normalization
    norm_percentiles: Tuple[float, float] = (2.0, 98.0)

    # Bead segmentation
    bead_blur_sigma: float = 2.0
    bead_closing_radius: int = 5
    bead_opening_radius: int = 3
    bead_hole_area: int = 5000  # "hole area threshold" (see compat wrapper)

    # Ag segmentation (top-hat + threshold)
    ag_tophat_radius: int = 7
    ag_min_object_size: int = 25  # "object min size" (see compat wrapper)
    ag_erode_bead_radius: int = 2


@dataclass
class AnalysisResult:
    """Holds outputs of a single analysis run."""
    image_path: Path
    raw: np.ndarray
    cropped: np.ndarray
    norm: np.ndarray
    bead_mask: np.ndarray
    ag_mask: np.ndarray
    tophat: np.ndarray
    ag_threshold: float
    coverage: float
    meta: dict[str, Any]


# -----------------------------
# Analyzer
# -----------------------------
class SEMCoverageAnalyzer:
    """
    Analyze SEM images of polystyrene beads decorated with Ag nanoparticles
    and estimate 2D projected coverage of Ag on the bead surface.

    The main output is:
        coverage = (Ag pixels inside bead ROI) / (bead ROI pixels)

    Parameters
    ----------
    config:
        AnalyzerConfig with pipeline parameters.
    logger:
        Optional logger. If not provided, a module-level logger is used.
    """

    def __init__(self, config: AnalyzerConfig = AnalyzerConfig(), logger: Optional[logging.Logger] = None):
        self.config = config
        self.log = logger or logging.getLogger(__name__)

    # ---------- Public API ----------
    def analyze(self, image_path: str | Path) -> AnalysisResult:
        """
        Run the full pipeline on a single TIFF image.

        Raises
        ------
        ImageLoadError, PreprocessError, SegmentationError
        """
        image_path = Path(image_path)

        raw = self._load_image(image_path)
        cropped, crop_row = self._crop_infobar(raw)
        norm = self._normalize(cropped)

        bead_mask = self._segment_bead(norm)
        ag_mask, tophat, thr = self._segment_ag(norm, bead_mask)

        cov = self._compute_coverage(bead_mask, ag_mask)

        # particle/cluster stats (connected components)
        ag_lab = label(ag_mask)
        ag_regions = regionprops(ag_lab)
        ag_count = int(ag_lab.max())
        ag_areas = np.array([r.area for r in ag_regions], dtype=np.int32) if ag_regions else np.array([], dtype=np.int32)

        meta = {
            "crop_row": int(crop_row),
            "shape_raw": tuple(raw.shape),
            "shape_cropped": tuple(cropped.shape),
            "dtype_raw": str(raw.dtype),
            "bead_area_px": int(bead_mask.sum()),
            "ag_area_px": int(ag_mask.sum()),
            "ag_count": ag_count,
            "ag_area_mean": float(ag_areas.mean()) if ag_areas.size else 0.0,
            "ag_area_median": float(np.median(ag_areas)) if ag_areas.size else 0.0,
            "config": self.config,
        }

        self.log.info(
            "Analyzed %s | coverage=%.4f | bead_px=%d | ag_px=%d | ag_count=%d",
            image_path.name, cov, meta["bead_area_px"], meta["ag_area_px"], meta["ag_count"]
        )

        return AnalysisResult(
            image_path=image_path,
            raw=raw,
            cropped=cropped,
            norm=norm,
            bead_mask=bead_mask,
            ag_mask=ag_mask,
            tophat=tophat,
            ag_threshold=float(thr),
            coverage=float(cov),
            meta=meta,
        )

    def show_debug(self, res: AnalysisResult, profile_image: str = "norm") -> None:
        """
        Show a diagnostic matplotlib figure:
        - normalized image
        - top-hat response
        - overlay with bead boundary (green) and Ag boundaries (red)
        - interactive line profile: click 2 points in the overlay panel

        Parameters
        ----------
        res:
            AnalysisResult returned by `analyze()`.
        profile_image:
            Which image to profile: "norm" or "tophat" or "cropped".
        """
        import matplotlib.pyplot as plt

        if profile_image not in {"norm", "tophat", "cropped"}:
            raise ValueError("profile_image must be one of: 'norm', 'tophat', 'cropped'")

        if profile_image == "norm":
            prof_img = res.norm
        elif profile_image == "tophat":
            prof_img = res.tophat
        else:
            # cropped is raw-like; normalize to [0,1] for nicer display/profile
            prof_img = self._normalize(res.cropped)

        overlay = self._make_overlay(res.norm, res.bead_mask, res.ag_mask)

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2)

        ax_img = fig.add_subplot(gs[0, 0])
        ax_th = fig.add_subplot(gs[0, 1])
        ax_ov = fig.add_subplot(gs[1, 0])
        ax_pr = fig.add_subplot(gs[1, 1])

        ax_img.set_title("Normalized")
        ax_img.imshow(res.norm, cmap="gray")
        ax_img.axis("off")

        ax_th.set_title(f"White top-hat (r={self.config.ag_tophat_radius})")
        ax_th.imshow(res.tophat, cmap="gray")
        ax_th.axis("off")

        ax_ov.set_title(
            f"Overlay | coverage={res.coverage:.3f} | ag_count={res.meta.get('ag_count', '?')}"
        )
        ax_ov.imshow(overlay)
        ax_ov.axis("off")

        ax_pr.set_title(f"Profile ({profile_image}) – click 2 points in overlay")
        ax_pr.set_xlabel("distance [px]")
        ax_pr.set_ylabel("intensity")
        (line_plot,) = ax_pr.plot([], [])
        ax_pr.grid(True, alpha=0.3)

        inter = _LineProfileInteractor(
            analyzer=self,
            fig=fig,
            ax_overlay=ax_ov,
            ax_profile=ax_pr,
            profile_line_artist=line_plot,
            img_for_profile=prof_img,
        )
        inter.connect()

        # Small help text
        fig.suptitle(
            "Controls: left-click 2 points = profile, right-click = reset, key 'r' = reset",
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def extract_profile(
            image: np.ndarray,
            p0: Tuple[float, float],
            p1: Tuple[float, float],
            linewidth: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract grayscale intensity profile along a line segment.

        Parameters
        ----------
        image:
            2D array (recommended: normalized float image).
        p0, p1:
            (row, col) coordinates of start and end.
        linewidth:
            Thickness (in pixels) to average across the line.

        Returns
        -------
        dist:
            Distance axis in pixels.
        values:
            Sampled intensity values along the line.
        """
        vals = profile_line(image, p0, p1, linewidth=linewidth, mode="reflect")
        dist = np.arange(vals.size, dtype=np.float32)
        return dist, vals.astype(np.float32)

    # ---------- IO ----------
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load a TIFF image and return a 2D numpy array."""
        try:
            img = imread(str(image_path))
        except Exception as e:
            raise ImageLoadError(f"Failed to load image '{image_path}': {e}") from e

        if img.ndim != 2:
            raise ImageLoadError(
                f"Unsupported image shape {img.shape} (ndim={img.ndim}). "
                "Expected a single-channel 2D TIFF."
            )

        self.log.debug("Loaded %s | shape=%s dtype=%s", image_path.name, img.shape, img.dtype)
        return img

    # ---------- Preprocessing ----------
    def _crop_infobar(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect and remove the SEM info bar at the bottom of the image.

        Strategy:
        - look at last N rows
        - compute row mean intensity
        - detect where row mean jumps above baseline + k*MAD for a minimum run

        Returns
        -------
        cropped_img, crop_row
            crop_row is the first excluded row index (i.e., img[:crop_row]).
        """
        cfg = self.config
        h = img.shape[0]
        tail = min(cfg.infobar_tail_rows, h)
        start = h - tail

        row_mean = img[start:].mean(axis=1).astype(np.float64)
        half = max(10, len(row_mean) // 2)

        baseline = np.median(row_mean[:half])
        mad = np.median(np.abs(row_mean[:half] - baseline)) + 1e-9
        thresh = baseline + cfg.infobar_k_mad * mad

        above = row_mean > thresh
        run = 0
        crop_row = h
        for i, flag in enumerate(above):
            run = run + 1 if flag else 0
            if run >= cfg.infobar_min_run:
                crop_row = start + (i - run + 1)
                break

        if crop_row == h:
            self.log.warning(
                "Info-bar not detected reliably; returning full image. Consider tuning infobar_*."
            )
        else:
            self.log.debug(
                "Info-bar detected: crop_row=%d (baseline=%.2f mad=%.2f thresh=%.2f)",
                crop_row, baseline, mad, thresh
            )

        if crop_row < 10:
            raise PreprocessError(f"Cropping removed too much of the image (crop_row={crop_row}).")

        return img[:crop_row, :], crop_row

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to float32 range [0, 1] using percentile scaling."""
        try:
            lo_p, hi_p = self.config.norm_percentiles
            lo, hi = np.percentile(img, (lo_p, hi_p))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                raise ValueError(f"Invalid percentiles: lo={lo}, hi={hi}")
            out = (img.astype(np.float32) - float(lo)) / (float(hi) - float(lo) + 1e-6)
            return np.clip(out, 0.0, 1.0)
        except Exception as e:
            raise PreprocessError(f"Normalization failed: {e}") from e

    # ---------- Segmentation ----------
    def _segment_bead(self, norm: np.ndarray) -> np.ndarray:
        """
        Segment the bead as the largest connected component after Otsu thresholding.

        Returns
        -------
        bead_mask : bool array
        """
        cfg = self.config
        try:
            blur = gaussian(norm, sigma=cfg.bead_blur_sigma, preserve_range=True)
            t = threshold_otsu(blur)
            mask = blur > t

            mask = closing(mask, disk(cfg.bead_closing_radius))
            mask = opening(mask, disk(cfg.bead_opening_radius))

            lab = label(mask)
            if lab.max() == 0:
                raise SegmentationError("Bead segmentation failed: no components found.")

            regions = regionprops(lab)
            largest = max(regions, key=lambda r: r.area)
            bead = (lab == largest.label)

            bead = remove_small_holes(bead, max_size=cfg.bead_hole_area - 1)

            if bead.sum() < 500:
                raise SegmentationError("Bead segmentation failed: ROI too small.")

            self.log.debug("Bead mask area (px): %d", int(bead.sum()))
            return bead
        except SEMAnalysisError:
            raise
        except Exception as e:
            raise SegmentationError(f"Bead segmentation failed: {e}") from e

    def _segment_ag(self, norm: np.ndarray, bead_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Segment Ag nanoparticles using white top-hat filtering and Otsu thresholding.

        Steps
        -----
        1) Erode bead mask slightly to avoid edge artifacts.
        2) Apply white top-hat to highlight small bright features.
        3) Compute Otsu threshold using pixels inside ROI only.
        4) Remove small objects and apply a tiny opening.

        Returns
        -------
        ag_mask, tophat_image, threshold
        """
        cfg = self.config
        try:
            roi = bead_mask.copy()
            if cfg.ag_erode_bead_radius > 0:
                roi = erosion(roi, disk(cfg.ag_erode_bead_radius))

            if roi.sum() < 500:
                raise SegmentationError("Ag segmentation failed: eroded ROI too small.")

            th = white_tophat(norm, footprint=disk(cfg.ag_tophat_radius)).astype(np.float32)

            vals = th[roi]
            if vals.size < 500:
                raise SegmentationError("Ag segmentation failed: insufficient ROI pixels for thresholding.")

            t = threshold_otsu(vals)
            ag = (th > t) & roi

            ag = remove_small_objects(ag, max_size=cfg.ag_min_object_size - 1)

            # a tiny opening to smooth single-pixel spikes
            ag = opening(ag, disk(1))

            self.log.debug("Ag mask area (px): %d | thr=%.6f", int(ag.sum()), float(t))
            return ag, th, float(t)
        except SEMAnalysisError:
            raise
        except Exception as e:
            raise SegmentationError(f"Ag segmentation failed: {e}") from e

    # ---------- Metrics ----------
    @staticmethod
    def _compute_coverage(bead_mask: np.ndarray, ag_mask: np.ndarray) -> float:
        """Compute coverage = Ag area / bead area."""
        denom = int(bead_mask.sum())
        if denom == 0:
            return float("nan")
        return float(int(ag_mask.sum()) / denom)

    # ---------- Visualization helpers ----------
    @staticmethod
    def _make_overlay(base_norm: np.ndarray, bead_mask: np.ndarray, ag_mask: np.ndarray) -> np.ndarray:
        """
        Create an RGB overlay:
        - grayscale base from base_norm
        - bead boundary in green
        - Ag boundaries in red
        """
        base = np.clip(base_norm, 0.0, 1.0).astype(np.float32)
        rgb = np.dstack([base, base, base])

        b_bead = find_boundaries(bead_mask, mode="outer")
        b_ag = find_boundaries(ag_mask, mode="outer")

        # green bead, red Ag
        rgb[b_bead] = (0.0, 1.0, 0.0)
        rgb[b_ag] = (1.0, 0.0, 0.0)
        return rgb


# -----------------------------
# Interactive profile helper
# -----------------------------
class _LineProfileInteractor:
    """
    Matplotlib interactor:
    - left click twice in overlay axis to define a segment
    - plot grayscale profile along the segment in profile axis
    - right click (or 'r') resets
    """

    def __init__(self, analyzer: SEMCoverageAnalyzer, fig, ax_overlay, ax_profile, profile_line_artist, img_for_profile):
        self.analyzer = analyzer
        self.fig = fig
        self.ax_overlay = ax_overlay
        self.ax_profile = ax_profile
        self.profile_line_artist = profile_line_artist
        self.img = img_for_profile

        self._p0: Optional[Tuple[float, float]] = None
        self._seg_artist = None

    def connect(self) -> None:
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _reset(self) -> None:
        self._p0 = None
        if self._seg_artist is not None:
            self._seg_artist.remove()
            self._seg_artist = None
        self.profile_line_artist.set_data([], [])
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "r":
            self._reset()

    def _on_click(self, event) -> None:
        # Only accept clicks inside overlay axes
        if event.inaxes != self.ax_overlay:
            return

        # Right click resets
        if event.button == 3:
            self._reset()
            return

        # Left click: collect two points
        if event.button != 1:
            return

        x, y = float(event.xdata), float(event.ydata)  # x=col, y=row
        if self._p0 is None:
            self._p0 = (y, x)
            return

        p1 = (y, x)
        p0 = self._p0
        self._p0 = None

        # draw segment in overlay
        if self._seg_artist is not None:
            self._seg_artist.remove()

        self._seg_artist = self.ax_overlay.plot([p0[1], p1[1]], [p0[0], p1[0]], "-", linewidth=2)[0]

        # compute profile
        dist, vals = self.analyzer.extract_profile(self.img, p0, p1, linewidth=1)
        self.profile_line_artist.set_data(dist, vals)

        self.ax_profile.relim()
        self.ax_profile.autoscale_view()
        self.fig.canvas.draw_idle()


# -----------------------------
# Logging helper
# -----------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
