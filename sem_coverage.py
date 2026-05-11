# sem_coverage_fast.py
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
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max


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
	  and enhance small bright features, then thresholds automatically.
	"""

	# Info-bar cropping
	infobar_tail_rows: int = 320
	infobar_k_mad: float = 8.0
	infobar_min_run: int = 10

	# Normalization (used for bead segmentation only)
	norm_percentiles: Tuple[float, float] = (2.0, 98.0)

	# Bead segmentation
	bead_blur_sigma: float = 2.0
	bead_closing_radius: int = 5
	bead_opening_radius: int = 3
	bead_hole_area: int = 5000  # remove holes < this (approx, see max_size-1)

	# Ag segmentation (fast + robust defaults)
	ag_tophat_radius: int = 7
	ag_min_object_size: int = 5      # <<< important: 25 often deletes everything after stricter threshold
	ag_erode_bead_radius: int = 2
	ag_use_log: bool = True          # log1p compresses dynamic range (often helps)

	# Counting via local maxima (fast alternative to watershed)
	count_min_distance: int = 5  # px
	count_thr_rel: float = 1.0   # threshold_abs = ag_threshold * count_thr_rel

	# Display
	display_percentiles: Tuple[float, float] = (0.5, 99.5)


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
	Analyze SEM images of beads decorated with Ag nanoparticles and estimate
	2D projected coverage of Ag on the bead surface.

		coverage = (Ag pixels inside bead ROI) / (bead ROI pixels)
	"""

	def __init__(self, config: AnalyzerConfig = AnalyzerConfig(), logger: Optional[logging.Logger] = None):
		self.config = config
		self.log = logger or logging.getLogger(__name__)

		# Precompute footprints (minor speed win, but free)
		self._fp_bead_close = disk(self.config.bead_closing_radius)
		self._fp_bead_open = disk(self.config.bead_opening_radius)
		self._fp_ag_roi_erode = disk(self.config.ag_erode_bead_radius) if self.config.ag_erode_bead_radius > 0 else None
		self._fp_ag_tophat = disk(self.config.ag_tophat_radius)
		self._fp_open1 = disk(1)

	# ---------- Public API ----------
	def analyze(self, image_path: str | Path) -> AnalysisResult:
		image_path = Path(image_path)

		raw = self._load_image(image_path)
		cropped, crop_row = self._crop_infobar(raw)
		norm = self._normalize(cropped)

		bead_mask = self._segment_bead(norm)

		# IMPORTANT: Ag segmentation runs on cropped raw-like intensities (not [0,1] normalized)
		ag_mask, tophat, thr = self._segment_ag(cropped, bead_mask)

		cov = self._compute_coverage(bead_mask, ag_mask)

		ag_count = self._count_ag_peaks(tophat, ag_mask, thr)
		# ag_areas = np.array([r.area for r in ag_regions], dtype=np.int32) if ag_regions else np.array([], dtype=np.int32)

		meta = {
			"crop_row": int(crop_row),
			"shape_raw": tuple(raw.shape),
			"shape_cropped": tuple(cropped.shape),
			"dtype_raw": str(raw.dtype),
			"bead_area_px": int(bead_mask.sum()),
			"ag_area_px": int(ag_mask.sum()),
			"ag_count": ag_count,
		 	# "ag_area_mean": float(ag_areas.mean()) if ag_areas.size else 0.0,
		 	# "ag_area_median": float(np.median(ag_areas)) if ag_areas.size else 0.0,
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
		Diagnostic matplotlib figure:
		- bead segmentation input (normalized)
		- Ag feature image (tophat, scaled for display)
		- overlay on original-like cropped image
		- interactive line profile: click 2 points in the overlay panel
		"""
		import matplotlib.pyplot as plt

		if profile_image not in {"norm", "tophat", "cropped"}:
			raise ValueError("profile_image must be one of: 'norm', 'tophat', 'cropped'")

		base_disp = self._scale_for_display(res.cropped)

		if profile_image == "norm":
			prof_img = res.norm
		elif profile_image == "tophat":
			prof_img = res.tophat
		else:
			prof_img = base_disp

		overlay = self._make_overlay(base_disp, res.bead_mask, res.ag_mask)

		fig = plt.figure(figsize=(14, 9))
		gs = fig.add_gridspec(2, 2)

		ax_img = fig.add_subplot(gs[0, 0])
		ax_th = fig.add_subplot(gs[0, 1])
		ax_ov = fig.add_subplot(gs[1, 0])
		ax_pr = fig.add_subplot(gs[1, 1])

		ax_img.set_title("Bead segmentation input (normalized)")
		ax_img.imshow(res.norm, cmap="gray")
		ax_img.axis("off")

		ax_th.set_title("Ag feature image (top-hat, scaled for display)")
		ax_th.imshow(self._scale_for_display(res.tophat), cmap="gray")
		ax_th.axis("off")

		ax_ov.set_title(
			f"Overlay on cropped | coverage={res.coverage:.3f} | ag_count={res.meta.get('ag_count', '?')}"
		)
		ax_ov.imshow(overlay)

		coords = peak_local_max(
			res.tophat,
			labels=res.ag_mask.astype(np.uint8),
			min_distance=int(self.config.count_min_distance),
			threshold_abs=float(res.ag_threshold) * float(self.config.count_thr_rel),
			exclude_border=False,
		)
		if coords.size:
			ax_ov.plot(coords[:, 1], coords[:, 0], "c.", markersize=4)  # cyan dots
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

		fig.suptitle("Controls: left-click 2 points = profile, right-click = reset, key 'r' = reset", fontsize=10)
		plt.tight_layout()
		plt.show()

	@staticmethod
	def extract_profile(
		image: np.ndarray,
		p0: Tuple[float, float],
		p1: Tuple[float, float],
		linewidth: int = 1,
	) -> Tuple[np.ndarray, np.ndarray]:
		vals = profile_line(image, p0, p1, linewidth=linewidth, mode="reflect")
		dist = np.arange(vals.size, dtype=np.float32)
		return dist, vals.astype(np.float32)

	# ---------- IO ----------
	def _load_image(self, image_path: Path) -> np.ndarray:
		try:
			img = imread(str(image_path))
		except Exception as e:
			raise ImageLoadError(f"Failed to load image '{image_path}': {e}") from e

		if img.ndim != 2:
			raise ImageLoadError(
				f"Unsupported image shape {img.shape} (ndim={img.ndim}). Expected a single-channel 2D TIFF."
			)

		self.log.debug("Loaded %s | shape=%s dtype=%s", image_path.name, img.shape, img.dtype)
		return img

	# ---------- Preprocessing ----------
	def _crop_infobar(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
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
			self.log.warning("Info-bar not detected reliably; returning full image. Consider tuning infobar_*.")
		else:
			self.log.debug(
				"Info-bar detected: crop_row=%d (baseline=%.2f mad=%.2f thresh=%.2f)",
				crop_row, baseline, mad, thresh
			)

		if crop_row < 10:
			raise PreprocessError(f"Cropping removed too much of the image (crop_row={crop_row}).")

		return img[:crop_row, :], crop_row

	def _normalize(self, img: np.ndarray) -> np.ndarray:
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
		cfg = self.config
		try:
			blur = gaussian(norm, sigma=cfg.bead_blur_sigma, preserve_range=True)
			t = threshold_otsu(blur)
			mask = blur > t

			mask = closing(mask, self._fp_bead_close)
			mask = opening(mask, self._fp_bead_open)

			lab = label(mask)
			if lab.max() == 0:
				raise SegmentationError("Bead segmentation failed: no components found.")

			regions = regionprops(lab)
			largest = max(regions, key=lambda r: r.area)
			bead = (lab == largest.label)

			# skimage>=0.26: max_size removes <= max_size, so use -1 to mimic old "< area_threshold"
			bead = remove_small_holes(bead, max_size=cfg.bead_hole_area - 1)

			if bead.sum() < 500:
				raise SegmentationError("Bead segmentation failed: ROI too small.")

			self.log.debug("Bead mask area (px): %d", int(bead.sum()))
			return bead
		except SEMAnalysisError:
			raise
		except Exception as e:
			raise SegmentationError(f"Bead segmentation failed: {e}") from e

	def _segment_ag(self, img: np.ndarray, bead_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
		"""
		Fast automatic Ag segmentation.

		- Uses log1p optionally (dynamic range compression).
		- Uses white top-hat with radius `ag_tophat_radius` (this is already a morphological
		  background suppression for structures larger than the radius).
		- Threshold is automatic (Otsu) computed ONLY within bead ROI.
		"""
		cfg = self.config
		try:
			roi = bead_mask.copy()
			if self._fp_ag_roi_erode is not None:
				roi = erosion(roi, self._fp_ag_roi_erode)

			if roi.sum() < 500:
				raise SegmentationError("Ag segmentation failed: eroded ROI too small.")

			img_f = img.astype(np.float32)
			if cfg.ag_use_log:
				img_f = np.log1p(img_f)

			feat = white_tophat(img_f, footprint=self._fp_ag_tophat).astype(np.float32)

			vals = feat[roi]
			if vals.size < 500:
				raise SegmentationError("Ag segmentation failed: insufficient ROI pixels for thresholding.")

			# Otsu can fail if vals are constant (rare). Guard it.
			if float(vals.max() - vals.min()) < 1e-12:
				t = float(vals.max()) + 1e-6
			else:
				t = float(threshold_otsu(vals))

			ag = (feat > t) & roi

			# skimage>=0.26: max_size removes <= max_size, so use -1 to mimic old "< min_size"
			ag = remove_small_objects(ag, max_size=cfg.ag_min_object_size - 1)
			ag = opening(ag, self._fp_open1)

			# extra debug stats (helps when something goes to zero)
			p50 = float(np.percentile(vals, 50))
			p90 = float(np.percentile(vals, 90))
			p99 = float(np.percentile(vals, 99))
			self.log.debug(
				"Ag feat stats in ROI: min=%.6g p50=%.6g p90=%.6g p99=%.6g max=%.6g | thr=%.6g | ag_px=%d",
				float(vals.min()), p50, p90, p99, float(vals.max()), float(t), int(ag.sum())
			)

			return ag, feat, float(t)
		except SEMAnalysisError:
			raise
		except Exception as e:
			raise SegmentationError(f"Ag segmentation failed: {e}") from e

	def _count_ag_peaks(self, feat: np.ndarray, mask: np.ndarray, thr: float) -> int:
		"""
		Count Ag 'particles' as the number of local maxima in the feature image.

		Parameters
		----------
		feat:
			Feature image (e.g., white top-hat result).
		mask:
			Boolean mask restricting where peaks are searched (e.g., ag_mask or ROI).
		thr:
			Base threshold (e.g., Otsu threshold) used to reject weak peaks.

		Returns
		-------
		int
			Number of detected peaks.
		"""
		cfg = self.config
		if mask.sum() == 0:
			return 0

		coords = peak_local_max(
			feat,
			labels=mask.astype(np.uint8),
			min_distance=int(cfg.count_min_distance),
			threshold_abs=float(thr) * float(cfg.count_thr_rel),
			exclude_border=False,
		)
		return int(coords.shape[0])

	# ---------- Metrics ----------
	@staticmethod
	def _compute_coverage(bead_mask: np.ndarray, ag_mask: np.ndarray) -> float:
		denom = int(bead_mask.sum())
		if denom == 0:
			return float("nan")
		return float(int(ag_mask.sum()) / denom)

	# ---------- Visualization helpers ----------
	@staticmethod
	def _make_overlay(base: np.ndarray, bead_mask: np.ndarray, ag_mask: np.ndarray) -> np.ndarray:
		base = np.clip(base, 0.0, 1.0).astype(np.float32)
		rgb = np.dstack([base, base, base])

		b_bead = find_boundaries(bead_mask, mode="outer")
		b_ag = find_boundaries(ag_mask, mode="outer")

		rgb[b_bead] = (0.0, 1.0, 0.0)
		rgb[b_ag] = (1.0, 0.0, 0.0)
		return rgb

	def _scale_for_display(self, img: np.ndarray) -> np.ndarray:
		lo_p, hi_p = self.config.display_percentiles
		lo, hi = np.percentile(img, (lo_p, hi_p))
		hi = max(hi, lo + 1e-6)
		out = (img.astype(np.float32) - float(lo)) / (float(hi) - float(lo))
		return np.clip(out, 0.0, 1.0)


# -----------------------------
# Interactive profile helper
# -----------------------------
class _LineProfileInteractor:
	"""Matplotlib interactor: left click twice in overlay axis defines profile line."""

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
		if event.inaxes != self.ax_overlay:
			return
		if event.button == 3:
			self._reset()
			return
		if event.button != 1:
			return

		x, y = float(event.xdata), float(event.ydata)  # x=col, y=row
		if self._p0 is None:
			self._p0 = (y, x)
			return

		p1 = (y, x)
		p0 = self._p0
		self._p0 = None

		if self._seg_artist is not None:
			self._seg_artist.remove()

		self._seg_artist = self.ax_overlay.plot([p0[1], p1[1]], [p0[0], p1[0]], "-", linewidth=2)[0]

		dist, vals = self.analyzer.extract_profile(self.img, p0, p1, linewidth=1)
		self.profile_line_artist.set_data(dist, vals)

		self.ax_profile.relim()
		self.ax_profile.autoscale_view()
		self.fig.canvas.draw_idle()


# -----------------------------
# Logging helper
# -----------------------------
def setup_logging(level: int = logging.INFO) -> None:
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
	)
