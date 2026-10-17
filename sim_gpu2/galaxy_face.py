import sys, os, glob, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1].rstrip("/")
half = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0      # half-width of the view, in disk scale lengths
files = sorted(glob.glob(f"{run}/pts_*.npz"))
frame_dir = f"{run}/face_on"
os.makedirs(frame_dir, exist_ok=True)
for i, f in enumerate(files):
    d = np.load(f)
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.set_facecolor("black")
    for key, color, label in (("neg", "#3b6cff", "twin matter"), ("pos", "#ffe14f", "ordinary matter")):
        p = d[key]
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], s=0.3, c=color, alpha=0.5, linewidths=0, label=label)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"t = {float(d['t']):.1f} rotations (face-on)", color="white")
    fig.savefig(f"{frame_dir}/frame_{i:04d}.png", dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)
out = f"/mnt/c/Users/roche/Desktop/{os.path.basename(run)}_face_on.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", f"{frame_dir}/frame_%04d.png",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
print("movie at", out)