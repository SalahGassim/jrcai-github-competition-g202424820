# RatSLAM USV — Python Port of OpenRatSLAM2

Single-file Python implementation of the bioinspired SLAM system described in:

> Coelho et al., "Bioinspired SLAM Approach for Unmanned Surface Vehicle",
> ICAR 2025 (OpenRatSLAM2)

The original is a ROS2 / C++ system. This port reads directly from a local
HDF5 dataset file — no ROS, no bag files.

---

## Quick Start

```bash
pip install -r requirements.txt

# Inspect HDF5 structure
python ratslam_usv.py --input dataset.h5 --probe

# Full run (live display)
python ratslam_usv.py --input dataset.h5 --output ./results/

# Headless (saves PNGs every 50 frames)
python ratslam_usv.py --input dataset.h5 --output ./results/ --headless

# Quick test on first 200 frames
python ratslam_usv.py --input dataset.h5 --max-frames 200 --headless
```

---

## Expected HDF5 Schema

| Path | Shape | dtype | Notes |
|------|-------|-------|-------|
| `/camera/image` | `(N, H, W, 3)` | uint8 | Frontal camera frames ~1 Hz |
| `/camera/timestamp` | `(N,)` | float64 | Seconds |
| `/odom/pose` | `(M, 3)` | float64 | `[x, y, theta]` absolute |
| `/odom/twist` | `(M, 2)` | float64 | `[v, omega]` (alternative) |
| `/odom/timestamp` | `(M,)` | float64 | |
| `/gps/lat_lon` | `(K, 2)` | float64 | WGS84, optional |
| `/gps/timestamp` | `(K,)` | float64 | |

If your file uses different paths, override with CLI flags:
```
--img-key /my/images  --odom-pose /my/odom  --gps-key /my/gps
```
Run `--probe` first to inspect the tree.

---

## Parameter Table (Table I from paper)

### Local View Cells

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image-crop-x-min` | 40 | Left crop pixel |
| `--image-crop-x-max` | 600 | Right crop pixel |
| `--image-crop-y-min` | 150 | Top crop pixel |
| `--image-crop-y-max` | 300 | Bottom crop pixel |
| `--template-x-size` | 60 | Template width (px) |
| `--template-y-size` | 20 | Template height (px) |
| `--vt-shift-match` | 25 | Max horizontal shift (px) |
| `--vt-step-match` | 5 | Step size for shift search |
| `--vt-match-threshold` | 0.073 | SAD threshold for new template |
| `--vt-patch-normalise` | 2 | Patch size for local normalisation |
| `--vt-active-decay` | 1.0 | Reactivation guard (frames) |

### Pose Cell Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pc-dim-xy` | 18 | XY dimension of 3D CAN |
| `--pc-dim-th` | 36 | θ dimension of 3D CAN |
| `--pc-cell-x-size` | 1.0 | Metres per cell |
| `--pc-vt-inject-energy` | 0.2 | δ — VT injection energy (Eq. 4) |
| `--exp-delta-pc-threshold` | 2.0 | New experience threshold |
| `--pc-vt-restore` | 0.05 | Restore weight after injection |

### Experience Map

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exp-loops` | 50 | Graph relaxation iterations |
| `--exp-initial-em-deg` | 180 | Initial heading (degrees) |
| `--exp-correction` | 0.5 | α — relaxation step weight (Eq. 8) |

---

## Outputs

All saved to `--output` directory (default `./results/`):

| File | Description |
|------|-------------|
| `trajectory_estimated.csv` | Experience nodes: `id,x,y,theta` |
| `trajectory_groundtruth.csv` | GPS track in local ENU: `id,east,north` |
| `experience_map.png` | Final map overlay |
| `timeline.png` | VT-id / experience-id vs frame (Fig. 6) |
| `final_pose_cells.png` | Pose cell activity heatmap |
| `hausdorff_report.txt` | Hausdorff distance vs GPS ground truth |

### Expected Performance

- **~16-minute trajectory** (≈900 m loop) processes offline in a few minutes on a modern laptop.
- **Hausdorff distance ~8 m** on the 900 m loop matches the paper's reported result.

---

## Algorithm Notes

The implementation follows Sections II–IV of the paper faithfully:

- **Pose Cell Network**: 3D continuous attractor network on an 18×18×36 torus.
  Circular convolution uses `scipy.signal.fftconvolve` with wrap-around padding
  (Eq. 1–3). Path integration uses sub-cell fractional shifting (Milford 2008).

- **Local View Cells**: SAD comparison with horizontal shift window for rotation
  robustness. Patch normalisation suppresses illumination variation.

- **Experience Map**: Topological graph with spring-based loop-closure relaxation
  (Eq. 5–8). Symmetric Hausdorff distance evaluation per Eq. (9).

---

## Dependencies

```
numpy scipy opencv-python h5py matplotlib tqdm pyproj
```

Python 3.10+ required. Install with `pip install -r requirements.txt`.
