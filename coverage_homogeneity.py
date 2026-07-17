"""Vectorized radial and polar coverage-homogeneity post-processing."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SegmentCoverage:
    index: int; inner: float; outer: float; center: float
    bead_pixel_count: int; ag_pixel_count: int; completeness: float; valid: bool
    projected_fraction: float | None; surface_weighted_fraction: float | None
    projected_over_cap_surface: float | None; surface_area_px2: float


@dataclass(frozen=True)
class HomogeneitySummary:
    valid_count: int; weighted_mean: float | None; sd_pp: float | None
    mad_pp: float | None; minimum: float | None; maximum: float | None
    range_pp: float | None; slope_pp_per_R: float | None = None


@dataclass(frozen=True)
class CoverageHomogeneityResult:
    rings: tuple[SegmentCoverage, ...]
    sectors: tuple[SegmentCoverage, ...]
    radial_summary: HomogeneitySummary
    polar_summary: HomogeneitySummary
    r_over_R: np.ndarray
    phi_rad: np.ndarray


def coordinate_maps(shape: tuple[int, int], center_rc: tuple[float, float], radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cached-compatible normalized radial and [0, 2*pi) angular maps."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    r = np.sqrt((rows-center_rc[0])**2 + (cols-center_rc[1])**2) / radius_px
    phi = np.mod(np.arctan2(rows-center_rc[0], cols-center_rc[1]), 2*math.pi)
    return r, phi


def _summary(items: list[SegmentCoverage], metric: str) -> HomogeneitySummary:
    valid = [item for item in items if item.valid and getattr(item, metric) is not None]
    if not valid: return HomogeneitySummary(0, None, None, None, None, None, None, None)
    v=np.asarray([getattr(i,metric) for i in valid],float); w=np.asarray([i.bead_pixel_count for i in valid],float)
    mean=float(np.average(v,weights=w)); sd=float(np.sqrt(np.average((v-mean)**2,weights=w)))*100
    mad=float(np.median(np.abs(v-np.median(v))))*100
    slope=float(np.polyfit([i.center for i in valid],v*100,1)[0]) if len(valid)>1 else None
    return HomogeneitySummary(len(valid),mean,sd,mad,float(v.min()),float(v.max()),float((v.max()-v.min())*100),slope)


def compute_homogeneity(bead: np.ndarray, ag: np.ndarray, center_rc: tuple[float,float], radius_px: float, *, inner: float, outer: float, width: float, sectors: int, min_completeness: float, metric: str) -> CoverageHomogeneityResult:
    """Compute disjoint rings and annular sectors directly from accepted masks."""
    if not (0 <= inner < outer <= 1 and 0 < width <= outer-inner and sectors >= 2 and 0 <= min_completeness <= 1):
        raise ValueError("Invalid homogeneity geometry.")
    bead=np.asarray(bead,bool); ag=np.asarray(ag,bool); r,phi=coordinate_maps(bead.shape,center_rc,radius_px)
    # Omit a final partial ring to keep widths comparable.
    n=int(math.floor((outer-inner)/width+1e-9)); edges=[inner+i*width for i in range(n+1)]
    def make(mask, idx, lo, hi, area):
        theoretical=int(mask.sum()); ref=mask&bead; bc=int(ref.sum()); ac=int((ref&ag).sum()); comp=bc/theoretical if theoretical else 0
        valid=bool(bc and comp>=min_completeness); denom=np.sqrt(np.maximum(radius_px**2-(r*radius_px)**2,1e-12)); weights=radius_px/denom
        return SegmentCoverage(idx,lo,hi,(lo+hi)/2,bc,ac,comp,valid,ac/bc if valid else None,float(weights[ref&ag].sum()/weights[ref].sum()) if valid else None,ac/area if valid and area>0 else None,area)
    rings=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(r>=lo)&((r<hi) if i<n-1 else (r<=hi)); area=2*math.pi*radius_px**2*(math.sqrt(1-lo*lo)-math.sqrt(1-hi*hi)); rings.append(make(mask,i,lo,hi,area))
    domain=(r>=inner)&(r<=outer); delta=2*math.pi/sectors; sector_items=[]
    zone=radius_px**2*(math.sqrt(1-inner*inner)-math.sqrt(1-outer*outer))
    for i in range(sectors):
        lo=i*delta; hi=(i+1)*delta; mask=domain&(phi>=lo)&((phi<hi) if i<sectors-1 else (phi<=hi)); sector_items.append(make(mask,i,lo,hi,delta*zone))
    return CoverageHomogeneityResult(tuple(rings),tuple(sector_items),_summary(rings,metric),_summary(sector_items,metric),r,phi)
