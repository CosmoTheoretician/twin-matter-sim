import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import glob

runs = [("out/lss_20261017_015917", "with twin matter"), ("out/lss_20261017_120618", "ordinary matter alone")]
z0, width, n = 0.5, 0.1, 256


def slab_map(p, box):
    p = np.mod(p, box)
    dz = (p[:, 2] / box - z0 + 0.5) % 1.0 - 0.5
    s = p[np.abs(dz) < width / 2]
    H, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=n, range=[[0, box], [0, box]])
    H = gaussian_filter(H, 1.0, mode="wrap")
    return np.log10(1 + H / max(H.mean(), 1e-12)).T


fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor="black")
d_mix = np.load(sorted(glob.glob(f"{runs[0][0]}/pts_*.npz"))[-1])
d_ctl = np.load(sorted(glob.glob(f"{runs[1][0]}/pts_*.npz"))[-1])
box = float(d_mix["box"])
panels = [(slab_map(d_mix["neg"], box), "twin matter (mixed run)", "Reds"),
          (slab_map(d_mix["pos"], box), "ordinary matter (mixed run)", "YlGnBu_r"),
          (slab_map(d_ctl["pos"], box), "ordinary matter ALONE (control)", "YlGnBu_r")]
vmax = max(panels[1][0].max(), panels[2][0].max())
for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=img.max() if "twin" in title else vmax,
              extent=[0, box, 0, box])
    ax.set_title(title, color="white", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"3-D, expansion from early on, t = {float(d_mix['t']):.0f} Jeans times, slab through the middle (10% of box)",
             color="white", fontsize=14)
out = "/mnt/c/Users/roche/Desktop/3d_control_comparison.png"
fig.savefig(out, dpi=110, facecolor="black", bbox_inches="tight")
print("saved", out)