"""
ratslam_usv.py — Single-file Python port of OpenRatSLAM2
"Bioinspired SLAM Approach for Unmanned Surface Vehicle" (Coelho et al., ICAR 2025)

Usage:
    python ratslam_usv.py --input dataset.h5 --output ./results/
    python ratslam_usv.py --input dataset.h5 --probe          # inspect HDF5 structure
    python ratslam_usv.py --input dataset.h5 --headless --max-frames 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gps_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Flat-earth approximation: WGS84 lat/lon → local ENU (east, north) in metres."""
    R = 6_378_137.0  # Earth radius (m)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = R * dlat
    east = R * math.cos(math.radians(lat0)) * dlon
    return east, north


def _hhmmss_to_seconds(t: int) -> float:
    """Convert HHMMSS-packed int (e.g. 125458 -> 12:54:58) to seconds since midnight."""
    t = int(t)
    h = t // 10000
    m = (t // 100) % 100
    s = t % 100
    return h * 3600 + m * 60 + s


def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two (N,2) arrays — Eq. (9)."""
    if len(A) == 0 or len(B) == 0:
        return float("inf")
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return max(d1, d2)


def _wrap_conv3d(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    3D circular (wrap-around) convolution via FFT — Eq. (1).
    Uses zero-padding then manual wrap-around to emulate mode='wrap'.
    """
    pad = [(s // 2, s // 2) for s in kernel.shape]
    padded = np.pad(data, pad, mode="wrap")
    result = fftconvolve(padded, kernel, mode="valid")
    # trim to original size
    slices = tuple(slice(0, s) for s in data.shape)
    return result[slices]


def _make_gaussian_kernel(sigma: float, size: int) -> np.ndarray:
    """1D Gaussian kernel, normalised."""
    x = np.arange(size) - size // 2
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


# ---------------------------------------------------------------------------
# 1. Pose Cell Network
# ---------------------------------------------------------------------------

class PoseCellNetwork:
    """
    3D continuous attractor network (CAN) of shape (dim_xy, dim_xy, dim_th).

    Axes: [x, y, theta]  — all wrap-around (torus topology).
    Implements excitation/inhibition per Eq. (1)–(3) and path integration
    with sub-cell fractional shifting per Milford (2008).
    """

    def __init__(self,
                 dim_xy: int = 18,
                 dim_th: int = 36,
                 pc_cell_x_size: float = 1.0,
                 vt_inject_energy: float = 0.2,
                 vt_restore: float = 0.05,
                 excite_sigma: float = 1.0,
                 inhibit_sigma: float = 2.0,
                 global_inhibit: float = 0.00002):
        self.dim_xy = dim_xy
        self.dim_th = dim_th
        self.pc_cell_x_size = pc_cell_x_size
        self.vt_inject_energy = vt_inject_energy
        self.vt_restore = vt_restore
        self.global_inhibit = global_inhibit  # φ

        # Activity tensor — starts as a single central bump
        self.P = np.zeros((dim_xy, dim_xy, dim_th), dtype=np.float64)
        cx, cy, ct = dim_xy // 2, dim_xy // 2, dim_th // 2
        self.P[cx, cy, ct] = 1.0

        # Precompute separable Gaussian kernels for excitation (Eq. 1–2)
        k_e_xy = _make_gaussian_kernel(excite_sigma, max(3, int(4 * excite_sigma) | 1))
        k_e_th = _make_gaussian_kernel(excite_sigma, max(3, int(4 * excite_sigma) | 1))
        # Build 3D excitation kernel (separable outer product)
        K2d = np.outer(k_e_xy, k_e_xy)
        self._excite_kernel = K2d[:, :, None] * k_e_th[None, None, :]

        # Inhibition kernel (broader Gaussian) — Eq. (3)
        k_i_xy = _make_gaussian_kernel(inhibit_sigma, max(3, int(6 * inhibit_sigma) | 1))
        k_i_th = _make_gaussian_kernel(inhibit_sigma, max(3, int(6 * inhibit_sigma) | 1))
        K2d_i = np.outer(k_i_xy, k_i_xy)
        self._inhibit_kernel = K2d_i[:, :, None] * k_i_th[None, None, :]

    # ------------------------------------------------------------------
    def excite(self) -> None:
        """Circular convolution with excitatory Gaussian — Eq. (1)."""
        self.P = _wrap_conv3d(self.P, self._excite_kernel)  # Eq. (1)

    def inhibit(self) -> None:
        """Local inhibition (subtract broader Gaussian) + global inhibition — Eq. (2)–(3)."""
        inhibited = _wrap_conv3d(self.P, self._inhibit_kernel)  # Eq. (2)
        self.P = self.P - inhibited * self.global_inhibit       # Eq. (3), φ term
        np.clip(self.P, 0.0, None, out=self.P)
        total = self.P.sum()
        if total > 1e-15:
            self.P /= total

    def path_integrate(self, vtrans: float, vrot: float, vrot_per_cell: float) -> None:
        """
        Shift activity packet by odometry increment (vtrans in cells, vrot in cells).
        Sub-cell interpolation distributes fractional shifts to neighbouring cells.
        Follows Milford (2008) fractional shift approach.
        """
        # Translational shift in pose-cell space (metres → cells)
        # We need current heading direction from the activity centroid
        cx, cy, ct = self.get_centroid()
        angle = (ct / self.dim_th) * 2 * math.pi  # radians

        dx_cells = vtrans * math.cos(angle) / self.pc_cell_x_size
        dy_cells = vtrans * math.sin(angle) / self.pc_cell_x_size
        dth_cells = vrot / vrot_per_cell

        self.P = _fractional_shift_3d(self.P, dx_cells, dy_cells, dth_cells)
        np.clip(self.P, 0.0, None, out=self.P)
        total = self.P.sum()
        if total > 1e-15:
            self.P /= total

    def inject(self, ix: int, iy: int, ith: int, energy: float) -> None:
        """Inject VT energy into a pose cell — Eq. (4)."""
        ix = int(ix) % self.dim_xy
        iy = int(iy) % self.dim_xy
        ith = int(ith) % self.dim_th
        self.P[ix, iy, ith] += energy  # Eq. (4)
        # Restore (blend with current state) to avoid over-correction
        self.P = (1.0 - self.vt_restore) * self.P + self.vt_restore * self.P
        np.clip(self.P, 0.0, None, out=self.P)
        total = self.P.sum()
        if total > 1e-15:
            self.P /= total

    def get_centroid(self) -> Tuple[float, float, float]:
        """
        Compute activity centroid with circular mean for the theta axis.
        Returns (cx, cy, cth) in cell coordinates.
        """
        # Marginalise to 1D
        px = self.P.sum(axis=(1, 2))
        py = self.P.sum(axis=(0, 2))
        pth = self.P.sum(axis=(0, 1))

        xs = np.arange(self.dim_xy)
        ys = np.arange(self.dim_xy)
        ths = np.arange(self.dim_th)

        cx = np.dot(xs, px) / (px.sum() + 1e-15)
        cy = np.dot(ys, py) / (py.sum() + 1e-15)

        # Circular mean for theta
        angles = ths * (2 * math.pi / self.dim_th)
        sin_sum = np.dot(np.sin(angles), pth)
        cos_sum = np.dot(np.cos(angles), pth)
        cth_rad = math.atan2(sin_sum, cos_sum)
        cth = (cth_rad % (2 * math.pi)) / (2 * math.pi) * self.dim_th

        return cx, cy, cth

    def activity_xy(self) -> np.ndarray:
        """Max-projection over theta for visualisation."""
        return self.P.max(axis=2)


def _fractional_shift_3d(P: np.ndarray,
                          dx: float, dy: float, dth: float) -> np.ndarray:
    """
    Distribute activity by fractional shifts in x, y, theta.
    For each axis: weight = 1-frac goes to floor cell, frac goes to ceil cell.
    """
    dim_xy, _, dim_th = P.shape

    # --- theta shift ---
    dth_floor = int(math.floor(dth)) % dim_th
    dth_frac = dth - math.floor(dth)
    P = (1.0 - dth_frac) * np.roll(P, dth_floor, axis=2) + \
        dth_frac          * np.roll(P, (dth_floor + 1) % dim_th, axis=2)

    # --- x shift ---
    dx_floor = int(math.floor(dx)) % dim_xy
    dx_frac = dx - math.floor(dx)
    P = (1.0 - dx_frac) * np.roll(P, dx_floor, axis=0) + \
        dx_frac          * np.roll(P, (dx_floor + 1) % dim_xy, axis=0)

    # --- y shift ---
    dy_floor = int(math.floor(dy)) % dim_xy
    dy_frac = dy - math.floor(dy)
    P = (1.0 - dy_frac) * np.roll(P, dy_floor, axis=1) + \
        dy_frac          * np.roll(P, (dy_floor + 1) % dim_xy, axis=1)

    return P


# ---------------------------------------------------------------------------
# 2. Local View Cells
# ---------------------------------------------------------------------------

@dataclass
class ViewTemplate:
    id: int
    data: np.ndarray          # (template_y_size, template_x_size) float32
    pc_centroid: Tuple[float, float, float]  # (cx, cy, cth) at creation time
    decay: float = 0.0        # vt_active_decay counter


class LocalViewCells:
    """
    Processes camera frames, compares to stored templates, and manages the
    view-template library. Implements SAD with horizontal shift matching.
    """

    def __init__(self,
                 image_crop_x_min: int = 40,
                 image_crop_x_max: int = 600,
                 image_crop_y_min: int = 150,
                 image_crop_y_max: int = 300,
                 template_x_size: int = 60,
                 template_y_size: int = 20,
                 vt_shift_match: int = 25,
                 vt_step_match: int = 5,
                 vt_match_threshold: float = 0.073,
                 vt_patch_normalise: int = 2,
                 vt_normalisation: int = 0,
                 vt_active_decay: float = 1.0):

        # Sanity-check crop bounds
        if image_crop_y_min > image_crop_y_max:
            image_crop_y_min, image_crop_y_max = image_crop_y_max, image_crop_y_min
            print("[VT] Warning: y_min > y_max — swapped to match paper convention.")

        self.crop_x_min = image_crop_x_min
        self.crop_x_max = image_crop_x_max
        self.crop_y_min = image_crop_y_min
        self.crop_y_max = image_crop_y_max
        self.tmpl_x = template_x_size
        self.tmpl_y = template_y_size
        self.shift = vt_shift_match
        self.step = vt_step_match
        self.threshold = vt_match_threshold
        self.patch_norm = vt_patch_normalise
        self.normalisation = vt_normalisation
        self.active_decay = vt_active_decay

        self.templates: list[ViewTemplate] = []
        self.current_id: int = -1
        self.current_diff: float = float("inf")

    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Crop, resize, grayscale, patch-normalise."""
        # Crop
        crop = frame[self.crop_y_min:self.crop_y_max,
                     self.crop_x_min:self.crop_x_max]
        if crop.size == 0:
            raise ValueError(
                f"Crop region is empty — check crop params "
                f"(x:{self.crop_x_min}-{self.crop_x_max}, "
                f"y:{self.crop_y_min}-{self.crop_y_max}) "
                f"vs image shape {frame.shape}"
            )
        # Grayscale
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if crop.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        else:
            gray = crop
        # Resize to template size
        resized = cv2.resize(gray, (self.tmpl_x, self.tmpl_y),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
        # Patch normalisation
        if self.patch_norm > 0:
            resized = self._patch_normalise(resized, self.patch_norm)
        return resized

    def _patch_normalise(self, img: np.ndarray, patch_size: int) -> np.ndarray:
        """Local mean/std normalisation over patch_size × patch_size windows."""
        h, w = img.shape
        out = np.zeros_like(img)
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                patch = img[y:y + patch_size, x:x + patch_size]
                mu = patch.mean()
                sigma = patch.std()
                out[y:y + patch_size, x:x + patch_size] = (patch - mu) / (sigma + 1e-8)
        return out

    def _sad_min_shift(self, tmpl: np.ndarray, query: np.ndarray) -> float:
        """
        Compute minimum SAD over horizontal circular shifts in [-shift, +shift]
        with step size `step`. Rotation-robust comparison.
        """
        min_sad = float("inf")
        n = query.shape[1]  # width
        for s in range(-self.shift, self.shift + 1, self.step):
            shifted = np.roll(query, s, axis=1)
            sad = np.abs(tmpl - shifted).sum() / (tmpl.size + 1e-15)
            if sad < min_sad:
                min_sad = sad
        return min_sad

    def process_frame(self,
                      frame: np.ndarray,
                      pc_centroid: Tuple[float, float, float]
                      ) -> Tuple[int, float, bool]:
        """
        Compare frame to all stored templates.
        Returns (matched_id, min_diff, is_new_template).
        Also updates decay state per vt_active_decay.
        """
        query = self._preprocess(frame)

        # Decay active templates to prevent spurious re-activation
        for t in self.templates:
            t.decay = max(0.0, t.decay - 1.0)

        if not self.templates:
            return self._create_template(query, pc_centroid)

        # Find best match
        best_id = -1
        best_diff = float("inf")
        for t in self.templates:
            # Skip recently activated templates (active decay)
            if t.decay > self.active_decay:
                continue
            diff = self._sad_min_shift(t.data, query)
            if diff < best_diff:
                best_diff = diff
                best_id = t.id

        self.current_diff = best_diff

        if best_diff > self.threshold:
            # New template
            return self._create_template(query, pc_centroid)

        # Match found
        self.templates[best_id].decay = self.active_decay
        self.current_id = best_id
        return best_id, best_diff, False

    def _create_template(self,
                         query: np.ndarray,
                         pc_centroid: Tuple[float, float, float]
                         ) -> Tuple[int, float, bool]:
        tid = len(self.templates)
        self.templates.append(ViewTemplate(id=tid, data=query.copy(),
                                           pc_centroid=pc_centroid,
                                           decay=self.active_decay))
        self.current_id = tid
        self.current_diff = 0.0
        return tid, 0.0, True

    def compare(self, frame: np.ndarray) -> Tuple[int, float]:
        """Convenience: compare without updating state (used externally)."""
        query = self._preprocess(frame)
        if not self.templates:
            return -1, float("inf")
        best_id, best_diff = -1, float("inf")
        for t in self.templates:
            diff = self._sad_min_shift(t.data, query)
            if diff < best_diff:
                best_diff, best_id = diff, t.id
        return best_id, best_diff

    @property
    def num_templates(self) -> int:
        return len(self.templates)


# ---------------------------------------------------------------------------
# 3. Experience Map
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    id: int
    pc: Tuple[float, float, float]  # pose cell centroid (cx, cy, cth)
    vt_id: int
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    links: list = field(default_factory=list)  # list of (exp_id, dx, dy, dtheta)


class ExperienceMap:
    """
    Topological experience map with graph relaxation for loop closure.
    Implements Eq. (5)–(8).
    """

    def __init__(self,
                 exp_delta_pc_threshold: float = 2.0,
                 exp_loops: int = 50,
                 exp_correction: float = 0.5,
                 mu_p: float = 1.0,
                 mu_v: float = 1.0,
                 initial_heading_deg: float = 180.0):
        self.delta_pc = exp_delta_pc_threshold
        self.exp_loops = exp_loops
        self.correction = exp_correction  # α in Eq. (8)
        self.mu_p = mu_p
        self.mu_v = mu_v
        self.initial_heading = math.radians(initial_heading_deg)

        self.experiences: list[Experience] = []
        self.current_id: int = -1
        self._accum_x: float = 0.0
        self._accum_y: float = 0.0
        self._accum_th: float = 0.0
        self._odom_delta: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    def _pc_distance(self,
                     pc1: Tuple[float, float, float],
                     pc2: Tuple[float, float, float],
                     dim_xy: int,
                     dim_th: int) -> float:
        """Wrapped distance in pose-cell space."""
        def wrap_dist(a, b, n):
            d = abs(a - b)
            return min(d, n - d)
        dx = wrap_dist(pc1[0], pc2[0], dim_xy)
        dy = wrap_dist(pc1[1], pc2[1], dim_xy)
        dth = wrap_dist(pc1[2], pc2[2], dim_th)
        return math.sqrt(dx**2 + dy**2 + dth**2)

    def find_match(self,
                   pc_centroid: Tuple[float, float, float],
                   vt_id: int,
                   dim_xy: int,
                   dim_th: int) -> Tuple[int, float]:
        """
        Score each experience: S = μ_p·||P-Pⁱ|| + μ_v·|V-Vⁱ| — Eq. (5).
        Returns (best_id, best_score).
        """
        best_id, best_score = -1, float("inf")
        for e in self.experiences:
            pc_dist = self._pc_distance(pc_centroid, e.pc, dim_xy, dim_th)
            vt_dist = float(abs(vt_id - e.vt_id))
            score = self.mu_p * pc_dist + self.mu_v * vt_dist  # Eq. (5)
            if score < best_score:
                best_score = score
                best_id = e.id
        return best_id, best_score

    def update(self,
               pc_centroid: Tuple[float, float, float],
               vt_id: int,
               odom_dx: float,
               odom_dy: float,
               odom_dth: float,
               dim_xy: int,
               dim_th: int) -> Tuple[int, bool]:
        """
        Advance experience map by one step.
        Returns (experience_id, loop_closed).
        """
        # Accumulate odometry since last experience creation — Eq. (6)
        self._accum_x += odom_dx
        self._accum_y += odom_dy
        self._accum_th += odom_dth

        if not self.experiences:
            # First experience
            e = Experience(id=0, pc=pc_centroid, vt_id=vt_id,
                           x=0.0, y=0.0,
                           theta=self.initial_heading)
            self.experiences.append(e)
            self.current_id = 0
            self._accum_x = self._accum_y = self._accum_th = 0.0
            return 0, False

        best_id, best_score = self.find_match(pc_centroid, vt_id, dim_xy, dim_th)

        loop_closed = False
        if best_score >= self.delta_pc:
            # Create new experience — Eq. (6)–(7)
            prev = self.experiences[self.current_id]
            new_x = prev.x + self._accum_x
            new_y = prev.y + self._accum_y
            new_th = prev.theta + self._accum_th
            eid = len(self.experiences)
            e = Experience(id=eid, pc=pc_centroid, vt_id=vt_id,
                           x=new_x, y=new_y, theta=new_th)
            self.experiences.append(e)

            # Store transition link from previous
            prev.links.append((eid, self._accum_x, self._accum_y, self._accum_th))
            e.links.append((self.current_id, -self._accum_x, -self._accum_y, -self._accum_th))

            self.current_id = eid
            self._accum_x = self._accum_y = self._accum_th = 0.0

        elif best_id != self.current_id:
            # Loop closure to an OLD experience — Eq. (8)
            loop_closed = True
            prev = self.experiences[self.current_id]
            matched = self.experiences[best_id]

            # Add link if not already present
            existing = {lnk[0] for lnk in prev.links}
            if best_id not in existing:
                prev.links.append((best_id, self._accum_x, self._accum_y, self._accum_th))
                matched.links.append((self.current_id,
                                      -self._accum_x, -self._accum_y, -self._accum_th))

            self.current_id = best_id
            self._accum_x = self._accum_y = self._accum_th = 0.0

            # Graph relaxation
            self.iterate_relaxation()

        return self.current_id, loop_closed

    def iterate_relaxation(self) -> None:
        """
        Spring-based graph relaxation for exp_loops iterations — Eq. (8).
        Correction weight α = exp_correction.
        """
        for _ in range(self.exp_loops):
            for e in self.experiences:
                for link in e.links:
                    nid, dx, dy, dth = link
                    neighbour = self.experiences[nid]
                    # Expected position of neighbour from this experience — Eq. (8)
                    ex_x = e.x + dx * math.cos(e.theta) - dy * math.sin(e.theta)
                    ex_y = e.y + dx * math.sin(e.theta) + dy * math.cos(e.theta)
                    ex_th = e.theta + dth
                    # Apply correction
                    neighbour.x += self.correction * (ex_x - neighbour.x)      # Eq. (8)
                    neighbour.y += self.correction * (ex_y - neighbour.y)
                    neighbour.theta += self.correction * (ex_th - neighbour.theta)

    def get_trajectory(self) -> np.ndarray:
        """Return (N, 4) array: [id, x, y, theta]."""
        if not self.experiences:
            return np.empty((0, 4))
        rows = [[e.id, e.x, e.y, e.theta] for e in self.experiences]
        return np.array(rows, dtype=np.float64)

    def create_experience(self, *args, **kwargs):
        """Alias kept for external compatibility."""
        pass


# ---------------------------------------------------------------------------
# 4. RatSLAM Orchestrator
# ---------------------------------------------------------------------------

class RatSLAM:
    """
    Orchestrates PoseCellNetwork, LocalViewCells, and ExperienceMap.
    step(image, odom) → experience_id
    """

    def __init__(self, args: argparse.Namespace):
        self.pcn = PoseCellNetwork(
            dim_xy=args.pc_dim_xy,
            dim_th=args.pc_dim_th,
            pc_cell_x_size=args.pc_cell_x_size,
            vt_inject_energy=args.pc_vt_inject_energy,
            vt_restore=args.pc_vt_restore,
        )
        self.vt = LocalViewCells(
            image_crop_x_min=args.image_crop_x_min,
            image_crop_x_max=args.image_crop_x_max,
            image_crop_y_min=args.image_crop_y_min,
            image_crop_y_max=args.image_crop_y_max,
            template_x_size=args.template_x_size,
            template_y_size=args.template_y_size,
            vt_shift_match=args.vt_shift_match,
            vt_step_match=args.vt_step_match,
            vt_match_threshold=args.vt_match_threshold,
            vt_patch_normalise=args.vt_patch_normalise,
            vt_normalisation=args.vt_normalisation,
            vt_active_decay=args.vt_active_decay,
        )
        self.em = ExperienceMap(
            exp_delta_pc_threshold=args.exp_delta_pc_threshold,
            exp_loops=args.exp_loops,
            exp_correction=args.exp_correction,
            initial_heading_deg=args.exp_initial_em_deg,
        )

        self.args = args
        self.dim_xy = args.pc_dim_xy
        self.dim_th = args.pc_dim_th
        self.vt_per_cell = (2 * math.pi) / args.pc_dim_th  # radians per theta cell

        # Odometry state
        self._prev_odom: Optional[np.ndarray] = None
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_th = 0.0

        # History for visualiser
        self.history_vt: list[int] = []
        self.history_exp: list[int] = []
        self.loop_closures: list[int] = []

    # ------------------------------------------------------------------
    def step(self,
             image: np.ndarray,
             odom: np.ndarray) -> int:
        """
        Process one frame.
        odom: [x, y, theta] absolute pose, or [v, omega] twist — handled below.
        Returns current experience id.
        """
        # --- Odometry delta ---
        if odom.shape[0] == 3:
            # Absolute pose — compute delta
            if self._prev_odom is None:
                self._prev_odom = odom.copy()
            raw_dx = odom[0] - self._prev_odom[0]
            raw_dy = odom[1] - self._prev_odom[1]
            raw_dth = odom[2] - self._prev_odom[2]
            self._prev_odom = odom.copy()
        else:
            # Twist [v, omega] — integrate
            v, omega = float(odom[0]), float(odom[1])
            dt = 1.0  # assume 1 s per frame; loader can pass dt
            raw_dx = v * math.cos(self._odom_th) * dt
            raw_dy = v * math.sin(self._odom_th) * dt
            raw_dth = omega * dt

        self._odom_x += raw_dx
        self._odom_y += raw_dy
        self._odom_th += raw_dth

        # Distance & rotation this step
        vtrans = math.sqrt(raw_dx**2 + raw_dy**2)
        vrot = raw_dth

        # --- PCN path integration ---
        self.pcn.excite()
        self.pcn.path_integrate(vtrans, vrot, self.vt_per_cell)
        self.pcn.inhibit()

        # --- Local view ---
        pc_centroid = self.pcn.get_centroid()
        vt_id, vt_diff, is_new = self.vt.process_frame(image, pc_centroid)

        # --- Inject VT energy into PCN — Eq. (4) ---
        if not is_new and len(self.vt.templates) > 0:
            tmpl = self.vt.templates[vt_id]
            cx, cy, cth = tmpl.pc_centroid
            self.pcn.inject(int(round(cx)), int(round(cy)), int(round(cth)),
                            self.args.pc_vt_inject_energy)  # Eq. (4)

        # --- Experience map ---
        exp_id, loop_closed = self.em.update(
            pc_centroid, vt_id,
            raw_dx, raw_dy, raw_dth,
            self.dim_xy, self.dim_th
        )

        if loop_closed:
            self.loop_closures.append(exp_id)

        self.history_vt.append(vt_id)
        self.history_exp.append(exp_id)

        return exp_id


# ---------------------------------------------------------------------------
# 5. HDF5 Data Loader
# ---------------------------------------------------------------------------

# Known field aliases
_IMG_CANDIDATES  = ["/camera/image", "/camera/images", "/image", "/camera/rgb"]
_IMG_TS_CANDS    = ["/camera/timestamp", "/camera/timestamps", "/camera/time"]
_ODOM_POSE_CANDS = ["/odom/pose", "/odom/poses", "/odometry/pose", "/odom/position"]
_ODOM_TWIST_CANDS= ["/odom/twist", "/odometry/twist", "/odom/velocity"]
_ODOM_TS_CANDS   = ["/odom/timestamp", "/odom/timestamps", "/odometry/time"]
_GPS_CANDS       = ["/gps/lat_lon", "/gps/latlon", "/gps/position", "/gps/coordinates"]
_GPS_TS_CANDS    = ["/gps/timestamp", "/gps/timestamps", "/gps/time"]


def _probe_h5(h5: h5py.File) -> None:
    """Recursively print all datasets in the HDF5 file."""
    print("\n=== HDF5 File Structure ===")

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            dt = obj.dtype
            if dt.names:  # compound / structured dtype — print each field on its own line
                print(f"  /{name}  shape={obj.shape}  (compound dtype, {len(dt.names)} fields)")
                for fname in dt.names:
                    fdtype = dt.fields[fname][0]
                    print(f"      {fname!r:50s} {fdtype}")
            else:
                print(f"  /{name:40s} shape={obj.shape}  dtype={dt}")

    h5.visititems(_visitor)
    print("===========================\n")


def _find_field(h5: h5py.File, candidates: list[str],
                override: Optional[str] = None) -> Optional[str]:
    """Return first existing path from candidates (or override if given)."""
    if override:
        if override in h5:
            return override
        raise KeyError(f"Specified field '{override}' not found in HDF5 file.")
    for c in candidates:
        if c in h5:
            return c
    return None


class HDF5DataLoader:
    """
    Probes an HDF5 file, locates camera / odometry / GPS datasets,
    and yields synchronised (image, odom, timestamp) tuples in order.
    """

    def __init__(self, path: str, args: argparse.Namespace):
        self.path = path
        self.args = args
        self._h5: Optional[h5py.File] = None

    def __enter__(self):
        self._h5 = h5py.File(self.path, "r")
        _probe_h5(self._h5)
        self._locate_fields()
        return self

    def __exit__(self, *_):
        if self._h5:
            self._h5.close()

    def _locate_fields(self) -> None:
        h5 = self._h5
        a = self.args

        self.img_key  = _find_field(h5, _IMG_CANDIDATES,   getattr(a, "img_key",  None))
        self.img_ts   = _find_field(h5, _IMG_TS_CANDS,     getattr(a, "img_ts",   None))
        self.odom_pose= _find_field(h5, _ODOM_POSE_CANDS,  getattr(a, "odom_pose",None))
        self.odom_twist=_find_field(h5, _ODOM_TWIST_CANDS, getattr(a, "odom_twist",None))
        self.odom_ts  = _find_field(h5, _ODOM_TS_CANDS,    getattr(a, "odom_ts",  None))
        self.gps_key  = _find_field(h5, _GPS_CANDS,        getattr(a, "gps_key",  None))
        self.gps_ts   = _find_field(h5, _GPS_TS_CANDS,     getattr(a, "gps_ts",   None))

        assert self.img_key,  ("No camera image dataset found. "
                               "Use --img-key to specify the path.")
        assert self.img_ts,   ("No camera timestamp dataset found. "
                               "Use --img-ts to specify the path.")

        odom_key = self.odom_pose or self.odom_twist
        assert odom_key, ("No odometry dataset found (tried pose and twist). "
                          "Use --odom-pose or --odom-twist to specify.")
        assert self.odom_ts, ("No odometry timestamp dataset found. "
                              "Use --odom-ts to specify.")

        self._odom_key = odom_key
        self._use_twist = (odom_key == self.odom_twist)

        print(f"[Loader] Image:    {self.img_key}  → timestamps: {self.img_ts}")
        print(f"[Loader] Odometry: {self._odom_key} ({'twist' if self._use_twist else 'pose'})"
              f" → timestamps: {self.odom_ts}")
        if self.gps_key:
            print(f"[Loader] GPS:      {self.gps_key} → timestamps: {self.gps_ts}")
        else:
            print("[Loader] GPS: not found (ground truth unavailable)")

    # ------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return self._h5[self.img_key].shape[0]

    def gps_enu(self) -> Optional[np.ndarray]:
        """Return GPS track as (K, 2) ENU array, or None."""
        if not self.gps_key:
            return None
        ll = self._h5[self.gps_key][:]
        if ll.shape[1] < 2:
            return None
        lat0, lon0 = ll[0, 0], ll[0, 1]
        enu = np.array([gps_to_enu(r[0], r[1], lat0, lon0) for r in ll])
        return enu

    def iterate(self,
                max_frames: Optional[int] = None
                ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, float]]:
        """
        Yield (frame_idx, image_uint8, odom_vec, timestamp) tuples
        synchronised by nearest-neighbour timestamp matching.
        """
        h5 = self._h5
        img_ts  = h5[self.img_ts][:]
        odom_ts = h5[self.odom_ts][:]
        odom_ds = h5[self._odom_key]

        n = len(img_ts)
        if max_frames:
            n = min(n, max_frames)

        for i in range(n):
            t = float(img_ts[i])
            # Nearest-neighbour match in odometry timestamps
            j = int(np.argmin(np.abs(odom_ts - t)))
            odom_vec = np.asarray(odom_ds[j], dtype=np.float64)

            img = np.asarray(h5[self.img_key][i], dtype=np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            yield i, img, odom_vec, t


# ---------------------------------------------------------------------------
# 5b. Compound HDF5 Data Loader  (single /data dataset, one row per timestep)
# ---------------------------------------------------------------------------

# Required field names inside the compound dtype
_COMPOUND_REQUIRED = [
    "Image",
    "Latitude",
    "Longitude",
    "Heading (degrees Magnetic)",
    "Yaw rate [degrees/s]",
    "Time",
]


class CompoundHDF5DataLoader:
    """
    Loader for HDF5 files whose entire payload lives in a single compound
    dataset (e.g. /data, shape (N,)) where every row contains all modalities
    already aligned to one timestep.

    Implements the same interface as HDF5DataLoader:
      __enter__ / __exit__, num_frames, gps_enu(), iterate()
    """

    def __init__(self, path: str, args: argparse.Namespace):
        self.path = path
        self.args = args
        self._h5: Optional[h5py.File] = None
        self._ds = None          # the compound dataset object
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._valid_mask: Optional[np.ndarray] = None  # (N,) bool — valid GPS rows

    # ------------------------------------------------------------------
    def __enter__(self):
        self._h5 = h5py.File(self.path, "r")
        _probe_h5(self._h5)
        self._locate_and_validate()
        return self

    def __exit__(self, *_):
        if self._h5:
            self._h5.close()

    def _locate_and_validate(self) -> None:
        data_key = getattr(self.args, "data_key", "/data")
        if data_key not in self._h5:
            available = []
            self._h5.visititems(
                lambda n, o: available.append(f"/{n}") if isinstance(o, h5py.Dataset) else None
            )
            raise KeyError(
                f"Compound dataset '{data_key}' not found in {self.path}.\n"
                f"Available datasets: {available}\n"
                f"Use --data-key to specify the correct path."
            )

        self._ds = self._h5[data_key]
        dt = self._ds.dtype

        if not dt.names:
            raise TypeError(
                f"Dataset '{data_key}' does not have a compound/structured dtype "
                f"(got {dt}). Use the standard --img-key / --odom-pose loader instead."
            )

        missing = [f for f in _COMPOUND_REQUIRED if f not in dt.names]
        if missing:
            raise KeyError(
                f"The compound dataset '{data_key}' is missing required fields: {missing}\n"
                f"Available fields: {list(dt.names)}"
            )

        print(f"[CompoundLoader] Dataset: '{data_key}'  rows={self._ds.shape[0]}")
        print(f"[CompoundLoader] Odometry source: {getattr(self.args, 'odom_source', 'gps_pose')}")

        # --- odom_source=gps_pose warning ---
        odom_source = getattr(self.args, "odom_source", "gps_pose")
        if odom_source == "gps_pose":
            print(
                "[CompoundLoader] WARNING: --odom-source=gps_pose derives odometry directly "
                "from GPS.\n"
                "  The estimated trajectory will closely mirror GPS ground truth, making the\n"
                "  Hausdorff comparison non-informative. Use this mode only to verify the\n"
                "  pipeline runs end-to-end, then switch to --odom-source imu_yaw_gps_speed."
            )

        # --- Time field sanity check (req. 7) ---
        times_raw = self._ds["Time"][:5]
        times_sec = [_hhmmss_to_seconds(v) for v in times_raw]
        print(f"\n[CompoundLoader] First 5 'Time' values (raw HHMMSS): {times_raw.tolist()}")
        print(f"[CompoundLoader] First 5 'Time' values (seconds):    {times_sec}")
        if len(times_sec) > 1:
            dts = []
            for i in range(len(times_sec) - 1):
                d = times_sec[i + 1] - times_sec[i]
                if d < 0:
                    d += 86400.0   # midnight rollover
                dts.append(d)
            print(f"[CompoundLoader] First {len(dts)} dt values (seconds):        "
                  f"{[f'{d:.3f}' for d in dts]}")
        print()

        # --- GPS validity mask (build once; reused by gps_enu and iterate) ---
        lats = self._ds["Latitude"][:]
        lons = self._ds["Longitude"][:]
        self._valid_mask = (
            np.isfinite(lats) & np.isfinite(lons) &
            ~((lats == 0.0) & (lons == 0.0))
        )
        first_valid = np.argmax(self._valid_mask)
        if not self._valid_mask[first_valid]:
            print("[CompoundLoader] WARNING: No valid GPS fixes found — GPS ENU unavailable.")
            self._lat0 = None
            self._lon0 = None
        else:
            self._lat0 = float(lats[first_valid])
            self._lon0 = float(lons[first_valid])
            print(f"[CompoundLoader] ENU origin: lat={self._lat0:.6f}, lon={self._lon0:.6f} "
                  f"(row {first_valid})")

        # --- Image crop sanity check (req. 6) ---
        first_img = np.asarray(self._ds["Image"][0], dtype=np.uint8)
        H, W = first_img.shape[:2]
        cx_max = self.args.image_crop_x_max
        cy_max = self.args.image_crop_y_max
        if cx_max > W or cy_max > H:
            # Compute a sensible suggestion from actual dimensions
            sx_min = W // 8
            sx_max = W - W // 8
            sy_min = H // 4
            sy_max = H - H // 4
            print(
                f"[CompoundLoader] WARNING: crop bounds (x_max={cx_max}, y_max={cy_max}) "
                f"exceed image size {W}×{H}.\n"
                f"  Suggested values for {W}×{H} images:\n"
                f"    --image-crop-x-min {sx_min} --image-crop-x-max {sx_max} "
                f"--image-crop-y-min {sy_min} --image-crop-y-max {sy_max}"
            )

    # ------------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return self._ds.shape[0]

    def gps_enu(self) -> Optional[np.ndarray]:
        """Return (N, 2) ENU array; rows where GPS is invalid contain NaN."""
        if self._lat0 is None:
            return None
        lats = self._ds["Latitude"][:]
        lons = self._ds["Longitude"][:]
        enu = np.full((len(lats), 2), np.nan, dtype=np.float64)
        for i in np.where(self._valid_mask)[0]:
            enu[i] = gps_to_enu(float(lats[i]), float(lons[i]), self._lat0, self._lon0)
        return enu

    def iterate(self,
                max_frames: Optional[int] = None
                ) -> Iterator[Tuple[int, np.ndarray, np.ndarray, float]]:
        """
        Yield (frame_idx, image_uint8, odom_vec, timestamp) in row order.
        odom_vec shape depends on --odom-source:
          gps_pose          → (3,)  [east, north, theta_rad]
          imu_yaw_gps_speed → (2,)  [v_m_s, omega_rad_s]
          imu_only          → (2,)  [v_m_s, omega_rad_s]
        """
        odom_source = getattr(self.args, "odom_source", "gps_pose")
        n = self.num_frames
        if max_frames:
            n = min(n, max_frames)

        prev_t: Optional[float] = None
        prev_enu: Optional[Tuple[float, float]] = None  # for imu_yaw_gps_speed
        integrated_v: float = 0.0                       # for imu_only

        for i in range(n):
            row = self._ds[i]

            # --- timestamp & dt ---
            # Time field is HHMMSS-packed (e.g. 125458 = 12:54:58); convert to seconds
            # so that per-minute rollovers (125459→125500 = +41 raw, +1 real) don't
            # corrupt dt-dependent odometry modes.
            raw_t = int(row["Time"])
            t = _hhmmss_to_seconds(raw_t)

            if prev_t is None:
                dt = 1.0
            else:
                dt = t - prev_t
                if dt < 0:              # midnight rollover (23:59:59 -> 00:00:00)
                    dt += 86400.0
                if dt <= 0 or dt > 60.0:  # guard against bad/missing samples
                    dt = 1.0
            prev_t = t

            # --- image ---
            image = np.asarray(row["Image"], dtype=np.uint8)

            # --- heading: compass bearing (CW from N) → math angle (CCW from E) ---
            heading_deg = float(row["Heading (degrees Magnetic)"])
            theta = math.radians(90.0 - heading_deg)

            # --- yaw rate ---
            yaw_rate_deg_s = float(row["Yaw rate [degrees/s]"])
            omega = math.radians(yaw_rate_deg_s)

            # --- build odometry vector ---
            if odom_source == "gps_pose":
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                if self._lat0 is not None and self._valid_mask[i]:
                    east, north = gps_to_enu(lat, lon, self._lat0, self._lon0)
                else:
                    east, north = (0.0, 0.0) if prev_enu is None else prev_enu
                odom_vec = np.array([east, north, theta], dtype=np.float64)

            elif odom_source == "imu_yaw_gps_speed":
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                if self._lat0 is not None and self._valid_mask[i]:
                    east, north = gps_to_enu(lat, lon, self._lat0, self._lon0)
                    if prev_enu is not None:
                        de = east - prev_enu[0]
                        dn = north - prev_enu[1]
                        v = math.sqrt(de**2 + dn**2) / max(dt, 1e-6)
                    else:
                        v = 0.0
                    prev_enu = (east, north)
                else:
                    v = 0.0  # no valid GPS — coast
                odom_vec = np.array([v, omega], dtype=np.float64)

            else:  # imu_only
                # Integrate forward acceleration (drifts without zero-velocity update)
                accel_x_g = float(row["Acceleration x, forward (G)"])
                accel_x_ms2 = accel_x_g * 9.80665
                integrated_v += accel_x_ms2 * dt
                integrated_v = max(0.0, integrated_v)  # clamp: USV can't go backwards easily
                odom_vec = np.array([integrated_v, omega], dtype=np.float64)

            yield i, image, odom_vec, t


# ---------------------------------------------------------------------------
# 6. Visualiser
# ---------------------------------------------------------------------------

class Visualizer:
    """
    Live or headless matplotlib display.
    4-panel layout:
      top-left:  current camera frame + template inset
      top-right: pose cell XY max-projection heatmap
      bot-left:  experience map + GPS ground truth
      bot-right: VT-id vs experience-id timeline (Fig. 6)
    """

    def __init__(self,
                 headless: bool = False,
                 save_interval: int = 50,
                 output_dir: str = "./results"):
        self.headless = headless
        self.save_interval = save_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if headless:
            matplotlib.use("Agg")

        self.fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.35, wspace=0.3)
        self.ax_cam   = self.fig.add_subplot(gs[0, 0])
        self.ax_pc    = self.fig.add_subplot(gs[0, 1])
        self.ax_map   = self.fig.add_subplot(gs[1, 0])
        self.ax_tl    = self.fig.add_subplot(gs[1, 1])

        self.ax_cam.set_title("Camera Frame")
        self.ax_pc.set_title("Pose Cells (XY max-proj)")
        self.ax_map.set_title("Experience Map")
        self.ax_tl.set_title("VT / Exp Timeline")

        self._cam_im   = None
        self._pc_im    = None
        self._gps_pts  = None
        self._frame_counter = 0

        if not headless:
            plt.ion()
            plt.show(block=False)

    # ------------------------------------------------------------------
    def update(self,
               frame: np.ndarray,
               template: np.ndarray,
               pcn: PoseCellNetwork,
               em: ExperienceMap,
               vt_history: list,
               exp_history: list,
               gps_enu: Optional[np.ndarray],
               loop_closures: list) -> None:

        self._frame_counter += 1

        # --- Top-left: camera + template inset ---
        self.ax_cam.clear()
        self.ax_cam.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                           if frame.shape[-1] == 3 else frame, aspect="auto")
        # Overlay template in lower-left corner
        h, w = frame.shape[:2]
        th, tw = template.shape[:2]
        inset = self.ax_cam.inset_axes([0.0, 0.0, tw / w, th / h])
        inset.imshow(template, cmap="gray", aspect="auto")
        inset.axis("off")
        self.ax_cam.set_title(f"Frame #{self._frame_counter}")
        self.ax_cam.axis("off")

        # --- Top-right: pose cells ---
        self.ax_pc.clear()
        pc_xy = pcn.activity_xy()
        self.ax_pc.imshow(pc_xy.T, origin="lower", cmap="hot", aspect="auto")
        cx, cy, cth = pcn.get_centroid()
        self.ax_pc.plot(cx, cy, "c+", markersize=12, markeredgewidth=2)
        self.ax_pc.set_title("Pose Cells (XY max-proj)")

        # --- Bottom-left: experience map ---
        self.ax_map.clear()
        traj = em.get_trajectory()
        if len(traj) > 1:
            self.ax_map.plot(traj[:, 1], traj[:, 2], "b.-", linewidth=0.8,
                             markersize=4, label="Estimated")
            # Highlight loop-closure nodes
            for lc in loop_closures[-5:]:
                if lc < len(traj):
                    self.ax_map.plot(traj[lc, 1], traj[lc, 2], "r*", markersize=10)
        # Mark current
        if len(traj) > 0 and em.current_id < len(traj):
            cur = traj[em.current_id]
            self.ax_map.plot(cur[1], cur[2], "go", markersize=8, label="Current")
        # GPS ground truth
        if gps_enu is not None:
            self.ax_map.plot(gps_enu[:, 0], gps_enu[:, 1], "r--",
                             linewidth=0.8, label="GPS GT")
        self.ax_map.set_title("Experience Map")
        self.ax_map.legend(fontsize=7, loc="upper left")
        self.ax_map.set_aspect("equal", adjustable="datalim")

        # --- Bottom-right: timeline (Fig. 6) ---
        self.ax_tl.clear()
        n = len(vt_history)
        xs = list(range(n))
        self.ax_tl.plot(xs, vt_history,  "b-", linewidth=0.7, label="VT id")
        self.ax_tl.plot(xs, exp_history, "r-", linewidth=0.7, label="Exp id")
        self.ax_tl.set_title("VT / Experience Timeline")
        self.ax_tl.set_xlabel("Frame")
        self.ax_tl.legend(fontsize=7)

        self.fig.canvas.draw()
        if not self.headless:
            try:
                plt.pause(0.001)
            except Exception:
                pass

        if self.headless and (self._frame_counter % self.save_interval == 0):
            fname = os.path.join(self.output_dir,
                                 f"frame_{self._frame_counter:05d}.png")
            self.fig.savefig(fname, dpi=80)

    # ------------------------------------------------------------------
    def save_outputs(self,
                     em: ExperienceMap,
                     gps_enu: Optional[np.ndarray],
                     pcn: PoseCellNetwork,
                     vt_history: list,
                     exp_history: list,
                     output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        # experience_map.png
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        traj = em.get_trajectory()
        if len(traj) > 1:
            ax2.plot(traj[:, 1], traj[:, 2], "b.-", linewidth=1, markersize=5,
                     label="Estimated")
        if gps_enu is not None:
            ax2.plot(gps_enu[:, 0], gps_enu[:, 1], "r--", linewidth=1,
                     label="GPS GT")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.legend()
        ax2.set_title("Experience Map (final)")
        fig2.savefig(os.path.join(output_dir, "experience_map.png"), dpi=120)
        plt.close(fig2)

        # timeline.png
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        xs = list(range(len(vt_history)))
        ax3.plot(xs, vt_history,  "b-", linewidth=0.7, label="VT id")
        ax3.plot(xs, exp_history, "r-", linewidth=0.7, label="Exp id")
        ax3.set_xlabel("Frame")
        ax3.set_title("VT / Experience Timeline")
        ax3.legend()
        fig3.savefig(os.path.join(output_dir, "timeline.png"), dpi=120)
        plt.close(fig3)

        # final_pose_cells.png
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        ax4.imshow(pcn.activity_xy().T, origin="lower", cmap="hot", aspect="auto")
        ax4.set_title("Final Pose Cell Activity (XY max-proj)")
        fig4.savefig(os.path.join(output_dir, "final_pose_cells.png"), dpi=120)
        plt.close(fig4)

        # trajectory CSVs
        if len(traj) > 0:
            np.savetxt(os.path.join(output_dir, "trajectory_estimated.csv"),
                       traj, delimiter=",",
                       header="id,x,y,theta", comments="")
        if gps_enu is not None:
            gt = np.column_stack([np.arange(len(gps_enu)), gps_enu])
            np.savetxt(os.path.join(output_dir, "trajectory_groundtruth.csv"),
                       gt, delimiter=",",
                       header="id,east,north", comments="")

        # Hausdorff report — Eq. (9)
        report_path = os.path.join(output_dir, "hausdorff_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            if gps_enu is not None and len(traj) > 1:
                est_xy = traj[:, 1:3]
                dh = hausdorff(gps_enu, est_xy)  # Eq. (9)
                f.write(f"Symmetric Hausdorff distance dH: {dh:.4f} m\n")
                dh_fwd = directed_hausdorff(gps_enu, est_xy)[0]
                dh_rev = directed_hausdorff(est_xy, gps_enu)[0]
                f.write(f"  Directed GPS->Est:  {dh_fwd:.4f} m\n")
                f.write(f"  Directed Est->GPS:  {dh_rev:.4f} m\n")
                f.write("\nNote: paper reports ~8 m on 900 m loop.\n")
            else:
                f.write("GPS ground truth not available - Hausdorff not computed.\n")

        print(f"\n[Output] Saved all results to: {output_dir}")
        with open(report_path, encoding="utf-8") as f:
            print(f.read())

    def close(self) -> None:
        if not self.headless:
            plt.ioff()
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# 7. Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RatSLAM USV — Python port of OpenRatSLAM2 (Coelho et al., ICAR 2025)"
    )

    # I/O
    p.add_argument("--input",  "-i", required=True,
                   help="Path to HDF5 dataset file")
    p.add_argument("--output", "-o", default="./results",
                   help="Output directory (default: ./results)")
    p.add_argument("--probe",  action="store_true",
                   help="Print HDF5 structure and exit")
    p.add_argument("--headless", action="store_true",
                   help="No live display; save PNG every --save-interval frames")
    p.add_argument("--save-interval", type=int, default=50,
                   help="Save PNG every N frames in headless mode")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (for quick tests)")

    # Compound-dataset loader (alternative to the default multi-stream loader)
    p.add_argument("--compound", action="store_true",
                   help="Use CompoundHDF5DataLoader (single compound dataset per row)")
    p.add_argument("--data-key", default="/data",
                   help="HDF5 path for the compound dataset (default: /data)")
    p.add_argument("--odom-source",
                   choices=["gps_pose", "imu_yaw_gps_speed", "imu_only"],
                   default="gps_pose",
                   help=("Odometry source for CompoundHDF5DataLoader. "
                         "gps_pose=sanity-check only (GPS→pose, Hausdorff meaningless); "
                         "imu_yaw_gps_speed=twist [v,ω] from GPS speed + IMU yaw; "
                         "imu_only=twist from integrated accel (drifts without ZUPT). "
                         "Default: gps_pose"))

    # HDF5 field overrides (standard multi-stream loader)
    p.add_argument("--img-key",    default=None, help="HDF5 path for images")
    p.add_argument("--img-ts",     default=None, help="HDF5 path for image timestamps")
    p.add_argument("--odom-pose",  default=None, help="HDF5 path for pose odometry [x,y,th]")
    p.add_argument("--odom-twist", default=None, help="HDF5 path for twist odometry [v,omega]")
    p.add_argument("--odom-ts",    default=None, help="HDF5 path for odometry timestamps")
    p.add_argument("--gps-key",    default=None, help="HDF5 path for GPS lat/lon")
    p.add_argument("--gps-ts",     default=None, help="HDF5 path for GPS timestamps")

    # Local View Cell parameters (Table I)
    lv = p.add_argument_group("Local View Cells")
    lv.add_argument("--image-crop-x-min",   type=int,   default=40)
    lv.add_argument("--image-crop-x-max",   type=int,   default=600)
    lv.add_argument("--image-crop-y-min",   type=int,   default=150)
    lv.add_argument("--image-crop-y-max",   type=int,   default=300)
    lv.add_argument("--template-x-size",    type=int,   default=60)
    lv.add_argument("--template-y-size",    type=int,   default=20)
    lv.add_argument("--vt-shift-match",     type=int,   default=25)
    lv.add_argument("--vt-step-match",      type=int,   default=5)
    lv.add_argument("--vt-match-threshold", type=float, default=0.073)
    lv.add_argument("--vt-patch-normalise", type=int,   default=2)
    lv.add_argument("--vt-normalisation",   type=int,   default=0)
    lv.add_argument("--vt-active-decay",    type=float, default=1.0)

    # Pose Cell Network parameters
    pc = p.add_argument_group("Pose Cell Network")
    pc.add_argument("--pc-dim-xy",             type=int,   default=18)
    pc.add_argument("--pc-dim-th",             type=int,   default=36)
    pc.add_argument("--pc-cell-x-size",        type=float, default=1.0)
    pc.add_argument("--pc-vt-inject-energy",   type=float, default=0.2)
    pc.add_argument("--exp-delta-pc-threshold",type=float, default=2.0)
    pc.add_argument("--pc-vt-restore",         type=float, default=0.05)

    # Experience Map parameters
    em = p.add_argument_group("Experience Map")
    em.add_argument("--exp-loops",          type=int,   default=50)
    em.add_argument("--exp-initial-em-deg", type=float, default=180.0)
    em.add_argument("--exp-correction",     type=float, default=0.5)

    return p


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # --probe mode: just print structure and exit
    if args.probe:
        with h5py.File(args.input, "r") as h5:
            _probe_h5(h5)
        if args.compound:
            print("Compound-dataset mode. Use --data-key to override the dataset path.")
        else:
            print("Use --img-key, --odom-pose (or --odom-twist), etc. to override field names.")
            print("Pass --compound to use the single-dataset compound loader instead.")
        sys.exit(0)

    # Initialise components
    rat = RatSLAM(args)

    Loader = CompoundHDF5DataLoader if args.compound else HDF5DataLoader
    with Loader(args.input, args) as loader:
        n_frames = loader.num_frames
        if args.max_frames:
            n_frames = min(n_frames, args.max_frames)
        gps_enu = loader.gps_enu()

        viz = Visualizer(
            headless=args.headless,
            save_interval=args.save_interval,
            output_dir=args.output,
        )

        print(f"\n[RatSLAM] Starting — {n_frames} frames to process.")
        t0 = time.time()

        try:
            for idx, image, odom, ts in tqdm(
                    loader.iterate(max_frames=args.max_frames),
                    total=n_frames,
                    desc="Processing",
                    unit="frame"):

                exp_id = rat.step(image, odom)

                # Visualise every frame (or every N in headless mode)
                if args.headless and idx % args.save_interval != 0:
                    continue

                # Get current template for inset
                cur_vt = rat.vt.templates[rat.vt.current_id] if rat.vt.templates else None
                tmpl_img = cur_vt.data if cur_vt is not None else np.zeros(
                    (args.template_y_size, args.template_x_size), dtype=np.float32)

                viz.update(
                    frame=image,
                    template=tmpl_img,
                    pcn=rat.pcn,
                    em=rat.em,
                    vt_history=rat.history_vt,
                    exp_history=rat.history_exp,
                    gps_enu=gps_enu,
                    loop_closures=rat.loop_closures,
                )

        except KeyboardInterrupt:
            print("\n[RatSLAM] Interrupted by user.")

        elapsed = time.time() - t0
        print(f"\n[RatSLAM] Done — {n_frames} frames in {elapsed:.1f}s "
              f"({n_frames / max(elapsed, 1):.1f} fps)")
        print(f"[RatSLAM] Templates: {rat.vt.num_templates}  "
              f"Experiences: {len(rat.em.experiences)}  "
              f"Loop closures: {len(rat.loop_closures)}")

        viz.save_outputs(
            em=rat.em,
            gps_enu=gps_enu,
            pcn=rat.pcn,
            vt_history=rat.history_vt,
            exp_history=rat.history_exp,
            output_dir=args.output,
        )
        viz.close()


if __name__ == "__main__":
    main()
