import sys
import os
import glob
import base64
import numpy as np
from scipy.ndimage import gaussian_filter

run = sys.argv[1].rstrip("/")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64          # grid cells per side
smooth = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5  # smoothing, in cells
files = sorted(glob.glob(f"{run}/pts_*.npz"))
if not files:
    sys.exit("No pts_*.npz files found.")
out_dir = "/mnt/c/Users/roche/Desktop/" + os.path.basename(run) + "_density"
os.makedirs(out_dir, exist_ok=True)


def b64(arr):
    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode()


def density(p, box):
    if len(p) == 0:
        return np.zeros((n, n, n), dtype=np.float32)
    H, _ = np.histogramdd(np.mod(p, box), bins=(n, n, n), range=[(0, box)] * 3)
    H = gaussian_filter(H, smooth, mode="wrap")
    return np.log10(1.0 + H / H.mean()).astype(np.float32)


entries = []
for f in files:
    d = np.load(f)
    box = float(d["box"])
    if d["pos"].shape[1] != 3:
        sys.exit("This script is for 3-D runs.")
    h = box / n
    fields = {"ordinary": density(d["pos"], box), "twin": density(d["neg"], box)}
    name = os.path.basename(f).replace("pts_", "density_").replace(".npz", ".vti")
    with open(os.path.join(out_dir, name), "w") as out:
        out.write('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        out.write(f'<ImageData WholeExtent="0 {n-1} 0 {n-1} 0 {n-1}" Origin="{h/2} {h/2} {h/2}" Spacing="{h} {h} {h}">\n')
        out.write(f'<Piece Extent="0 {n-1} 0 {n-1} 0 {n-1}">\n<PointData Scalars="twin">\n')
        for key, arr in fields.items():
            out.write(f'<DataArray type="Float32" Name="{key}" format="binary">{b64(arr.ravel(order="F"))}</DataArray>\n')
        out.write('</PointData>\n</Piece>\n</ImageData>\n</VTKFile>\n')
    entries.append((float(d["t"]), name))

with open(os.path.join(out_dir, "density.pvd"), "w") as out:
    out.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, name in entries:
        out.write(f'  <DataSet timestep="{t}" file="{name}"/>\n')
    out.write('</Collection>\n</VTKFile>\n')
print(f"wrote {len(entries)} frames to {out_dir} (open density.pvd in ParaView)")