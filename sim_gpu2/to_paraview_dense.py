import sys
import os
import glob
import base64
import numpy as np
from scipy.ndimage import gaussian_filter

run = sys.argv[1].rstrip("/")
keep_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10   # fraction of densest particles to keep
n, smooth = 64, 1.5
files = sorted(glob.glob(f"{run}/pts_*.npz"))
if not files:
    sys.exit("No pts_*.npz files found.")
out_dir = "/mnt/c/Users/roche/Desktop/" + os.path.basename(run) + "_dense"
os.makedirs(out_dir, exist_ok=True)


def b64(arr):
    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode()


def dense(p, box):
    p = np.mod(p, box)
    H, _ = np.histogramdd(p, bins=(n,) * 3, range=[(0, box)] * 3)
    H = gaussian_filter(H, smooth, mode="wrap") / H.mean()
    i = np.minimum((p / box * n).astype(int), n - 1)
    rho = H[i[:, 0], i[:, 1], i[:, 2]]
    keep = rho >= np.quantile(rho, 1 - keep_frac)
    return p[keep], np.log10(1 + rho[keep])


entries = []
for f in files:
    d = np.load(f)
    box = float(d["box"])
    pa, ra = dense(d["pos"], box)
    pb, rb = dense(d["neg"], box)
    pts = np.concatenate([pa, pb]).astype(np.float32)
    species = np.concatenate([np.ones(len(pa)), -np.ones(len(pb))]).astype(np.int8)
    rho = np.concatenate([ra, rb]).astype(np.float32)
    m = len(pts)
    name = os.path.basename(f).replace("pts_", "dense_").replace(".npz", ".vtp")
    with open(os.path.join(out_dir, name), "w") as o:
        o.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        o.write(f'<PolyData><Piece NumberOfPoints="{m}" NumberOfVerts="{m}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        o.write('<PointData Scalars="species">\n')
        o.write(f'<DataArray type="Int8" Name="species" format="binary">{b64(species)}</DataArray>\n')
        o.write(f'<DataArray type="Float32" Name="density" format="binary">{b64(rho)}</DataArray>\n')
        o.write('</PointData>\n<Points>\n')
        o.write(f'<DataArray type="Float32" NumberOfComponents="3" format="binary">{b64(pts)}</DataArray>\n')
        o.write('</Points>\n<Verts>\n')
        o.write(f'<DataArray type="Int32" Name="connectivity" format="binary">{b64(np.arange(m, dtype=np.int32))}</DataArray>\n')
        o.write(f'<DataArray type="Int32" Name="offsets" format="binary">{b64(np.arange(1, m + 1, dtype=np.int32))}</DataArray>\n')
        o.write('</Verts>\n</Piece></PolyData>\n</VTKFile>\n')
    entries.append((float(d["t"]), name))

with open(os.path.join(out_dir, "dense.pvd"), "w") as o:
    o.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, name in entries:
        o.write(f'  <DataSet timestep="{t}" file="{name}"/>\n')
    o.write('</Collection>\n</VTKFile>\n')
print(f"wrote {len(entries)} frames to {out_dir} (open dense.pvd in ParaView)")
