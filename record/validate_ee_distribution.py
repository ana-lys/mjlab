"""Validate end-effector distribution from ffw_records.npz.

Produces:
  1. Per-axis histograms for left & right hand EE positions
  2. 2D workspace heatmaps (XY, XZ, YZ projections)
  3. Pair distance distribution (how far apart are start↔target EE)
  4. Coverage stats: bounding box, sparsity, per-bin counts
  5. Left vs right hand symmetry check

Saves all figures to record/figs/ and prints summary stats.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT_DIR = "/puffertank/mjlab/record/figs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading records...")
data = dict(np.load("/puffertank/mjlab/record/ffw_records.npz"))
N = data["target_ee_pos"].shape[0]
print(f"Total records: {N:,}\n")

# EE positions: [N, 2, 3]  (hand 0 = left, hand 1 = right)
# We have two sets: "target" (previous valid config) and "tobe" (this episode's start)
target_ee = data["target_ee_pos"]    # [N, 2, 3]
tobe_ee = data["target_tobe_ee_pos"] # [N, 2, 3]

# Combine both for overall workspace coverage analysis
all_ee = np.concatenate([target_ee, tobe_ee], axis=0)  # [2N, 2, 3]

HAND_NAMES = ["Left Hand", "Right Hand"]
AXIS_NAMES = ["X", "Y", "Z"]

# =============================================================================
# 1. SUMMARY STATS
# =============================================================================
print("=" * 60)
print("EE POSITION STATISTICS (all configs, both target & tobe)")
print("=" * 60)
for h, hname in enumerate(HAND_NAMES):
    pos = all_ee[:, h, :]  # [2N, 3]
    print(f"\n  {hname}:")
    for a, aname in enumerate(AXIS_NAMES):
        vals = pos[:, a]
        print(f"    {aname}: min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"median={np.median(vals):.4f}")

# =============================================================================
# 2. PER-AXIS HISTOGRAMS (left vs right, target vs tobe)
# =============================================================================
print("\nGenerating per-axis histograms...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("EE Position Distribution per Axis", fontsize=16)

for h in range(2):
    for a in range(3):
        ax = axes[h, a]
        ax.hist(target_ee[:, h, a], bins=200, alpha=0.5, label="target", density=True, color="tab:blue")
        ax.hist(tobe_ee[:, h, a], bins=200, alpha=0.5, label="tobe", density=True, color="tab:orange")
        ax.set_title(f"{HAND_NAMES[h]} — {AXIS_NAMES[a]}")
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_per_axis_histograms.png", dpi=150)
plt.close()

# =============================================================================
# 3. 2D WORKSPACE HEATMAPS (XY, XZ, YZ) per hand
# =============================================================================
print("Generating 2D workspace heatmaps...")
PROJECTIONS = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("EE Workspace Coverage (2D Projections, log scale)", fontsize=16)

for h in range(2):
    pos = all_ee[:, h, :]
    for p, (xname, yname, xi, yi) in enumerate(PROJECTIONS):
        ax = axes[h, p]
        hh = ax.hist2d(pos[:, xi], pos[:, yi], bins=150, norm=LogNorm(), cmap="viridis")
        ax.set_title(f"{HAND_NAMES[h]} — {xname} vs {yname}")
        ax.set_xlabel(f"{xname} (m)")
        ax.set_ylabel(f"{yname} (m)")
        ax.set_aspect("equal")
        plt.colorbar(hh[3], ax=ax, label="count (log)")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_workspace_heatmaps.png", dpi=150)
plt.close()

# =============================================================================
# 4. PAIR DISTANCE DISTRIBUTION (target ↔ tobe EE distance)
# =============================================================================
print("Computing pair distances...")
# Per-hand Euclidean distance between target and tobe
dist_per_hand = np.linalg.norm(target_ee - tobe_ee, axis=-1)  # [N, 2]
dist_total = np.linalg.norm(
    (target_ee - tobe_ee).reshape(N, -1), axis=-1
)  # [N] combined 6D distance

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Start ↔ Target EE Distance Distribution", fontsize=14)

for h in range(2):
    ax = axes[h]
    d = dist_per_hand[:, h]
    ax.hist(d, bins=200, density=True, color="tab:green", alpha=0.7)
    ax.axvline(d.mean(), color="red", linestyle="--", label=f"mean={d.mean():.3f}")
    ax.axvline(np.median(d), color="blue", linestyle="--", label=f"median={np.median(d):.3f}")
    ax.set_title(f"{HAND_NAMES[h]} distance")
    ax.set_xlabel("Euclidean distance (m)")
    ax.set_ylabel("Density")
    ax.legend()

ax = axes[2]
ax.hist(dist_total, bins=200, density=True, color="tab:purple", alpha=0.7)
ax.axvline(dist_total.mean(), color="red", linestyle="--", label=f"mean={dist_total.mean():.3f}")
ax.axvline(np.median(dist_total), color="blue", linestyle="--", label=f"median={np.median(dist_total):.3f}")
ax.set_title("Combined 6D distance")
ax.set_xlabel("Euclidean distance (m)")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_pair_distance.png", dpi=150)
plt.close()

print(f"\n{'=' * 60}")
print("PAIR DISTANCE STATS")
print(f"{'=' * 60}")
for h, hname in enumerate(HAND_NAMES):
    d = dist_per_hand[:, h]
    print(f"  {hname}: min={d.min():.4f}  max={d.max():.4f}  "
          f"mean={d.mean():.4f}  std={d.std():.4f}  median={np.median(d):.4f}")
    pcts = np.percentile(d, [5, 25, 50, 75, 95])
    print(f"    percentiles [5,25,50,75,95]: {pcts}")
    # Trivial/extreme pair counts
    trivial = (d < 0.02).sum()
    extreme = (d > 1.0).sum()
    print(f"    trivial (<2cm): {trivial:,} ({100*trivial/N:.2f}%)")
    print(f"    extreme (>1m):  {extreme:,} ({100*extreme/N:.2f}%)")

# =============================================================================
# 5. 3D BINNED COVERAGE ANALYSIS
# =============================================================================
print(f"\n{'=' * 60}")
print("3D BINNED COVERAGE (per hand)")
print(f"{'=' * 60}")

BIN_SIZE = 0.05  # 5cm bins

for h, hname in enumerate(HAND_NAMES):
    pos = all_ee[:, h, :]
    # Discretize into 5cm bins
    bins = np.floor(pos / BIN_SIZE).astype(np.int32)
    unique_bins = np.unique(bins, axis=0)
    n_occupied = len(unique_bins)

    # Bounding box in bin space
    bmin = bins.min(axis=0)
    bmax = bins.max(axis=0)
    grid_shape = bmax - bmin + 1
    n_total_bins = int(np.prod(grid_shape))
    coverage_pct = 100.0 * n_occupied / n_total_bins

    print(f"\n  {hname} ({BIN_SIZE*100:.0f}cm bins):")
    print(f"    Bounding box: X=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}] "
          f"Y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}] "
          f"Z=[{pos[:,2].min():.3f}, {pos[:,2].max():.3f}]")
    print(f"    Grid shape: {grid_shape}")
    print(f"    Total bins in bbox: {n_total_bins:,}")
    print(f"    Occupied bins: {n_occupied:,}")
    print(f"    Coverage: {coverage_pct:.1f}%")

    # Per-bin count distribution
    # Use a dict for sparse counting
    bin_keys = [tuple(b) for b in bins]
    from collections import Counter
    counts = Counter(bin_keys)
    count_vals = np.array(list(counts.values()))
    print(f"    Samples per occupied bin: min={count_vals.min()}  max={count_vals.max()}  "
          f"mean={count_vals.mean():.0f}  median={np.median(count_vals):.0f}")
    sparse = (count_vals < 10).sum()
    print(f"    Sparse bins (<10 samples): {sparse:,} ({100*sparse/n_occupied:.1f}% of occupied)")

# =============================================================================
# 6. LEFT vs RIGHT SYMMETRY CHECK
# =============================================================================
print(f"\n{'=' * 60}")
print("LEFT vs RIGHT HAND SYMMETRY")
print(f"{'=' * 60}")
left = all_ee[:, 0, :]
right = all_ee[:, 1, :]
# For a symmetric bimanual robot, right_Y ≈ -left_Y, and X/Z should be similar
print(f"  Left  mean: X={left[:,0].mean():.4f}  Y={left[:,1].mean():.4f}  Z={left[:,2].mean():.4f}")
print(f"  Right mean: X={right[:,0].mean():.4f}  Y={right[:,1].mean():.4f}  Z={right[:,2].mean():.4f}")
print(f"  Y-axis mirror check: mean(left_Y + right_Y) = {(left[:,1] + right[:,1]).mean():.4f}  "
      f"(should be ~0 for perfect symmetry)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Left vs Right Hand Symmetry", fontsize=14)
for a in range(3):
    ax = axes[a]
    ax.hist(left[:, a], bins=200, alpha=0.5, label="Left", density=True)
    if a == 1:  # Y axis: mirror right
        ax.hist(-right[:, a], bins=200, alpha=0.5, label="Right (mirrored -Y)", density=True)
    else:
        ax.hist(right[:, a], bins=200, alpha=0.5, label="Right", density=True)
    ax.set_title(f"{AXIS_NAMES[a]} axis")
    ax.set_xlabel("Position (m)")
    ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_symmetry.png", dpi=150)
plt.close()

# =============================================================================
# 7. BOOTSTRAP vs REAL CHECK (detect zero/default records)
# =============================================================================
print(f"\n{'=' * 60}")
print("DATA QUALITY — BOOTSTRAP/DEFAULT DETECTION")
print(f"{'=' * 60}")
target_jp = data["target_joint_pos"]
zero_joint = np.all(target_jp == 0, axis=1).sum()
default_ee = np.all(np.abs(target_ee[:, 0, :] - [0.5, 0.3, 1.0]) < 1e-3, axis=1).sum()
print(f"  Records with all-zero target_joint_pos: {zero_joint:,} ({100*zero_joint/N:.2f}%)")
print(f"  Records with default target_ee_pos [0.5,0.3,1.0]: {default_ee:,} ({100*default_ee/N:.2f}%)")
print(f"  → These are bootstrap records from the first episode (before any valid target existed)")

print(f"\n{'=' * 60}")
print(f"All figures saved to {OUT_DIR}/")
print(f"{'=' * 60}")
