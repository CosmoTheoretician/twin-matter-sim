import sys
import os
import glob
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1].rstrip("/")
z0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5        # slab centre, fraction of box
width = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05    # slab thickness, fraction of box
files = sorted(glob.glob(f"{run}/pts_*.npz"))
if not files:
    sys.exit("No pts_*.npz files found.")
tag = f"slab_z{z0:.2f}_w{width:.2f}"
frame_dir = f"{run}/{tag}"
os.makedirs(frame_dir, exist_ok=True)

for i, f in enumerate(files):
    d = np.load(f)
    box = float(d["box"])
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")
    for key, color, label in (("pos", "#ffe14f", "ordinary matter"), ("neg", "#3b6cff", "twin matter")):
        p = np.mod(d[key], box)
        dz = (p[:, 2] / box - z0 + 0.5) % 1.0 - 0.5     # periodic distance from the slab centre
        s = p[np.abs(dz) < width / 2]
        ax.scatter(s[:, 0], s[:, 1], s=0.6, c=color, alpha=0.7, linewidths=0, label=f"{label} ({len(s):,})")
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {float(d['t']):.1f}   slab z = {z0:.2f} ± {width/2:.3f} of box", color="white")
    leg = ax.legend(loc="upper right", markerscale=12, facecolor="black", labelcolor="white", fontsize=9)
    fig.savefig(f"{frame_dir}/frame_{i:04d}.png", dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)

out = f"/mnt/c/Users/roche/Desktop/{os.path.basename(run)}_{tag}.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", f"{frame_dir}/frame_%04d.png",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
print(f"frames in {frame_dir}, movie at {out}")
