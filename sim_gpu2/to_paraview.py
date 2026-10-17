import sys
import os
import glob
import base64
import numpy as np

run = sys.argv[1].rstrip("/")
files = sorted(glob.glob(f"{run}/pts_*.npz"))
if not files:
    sys.exit("No pts_*.npz files found; was the run made after the patch?")
out_dir = "/mnt/c/Users/roche/Desktop/" + os.path.basename(run) + "_paraview"
os.makedirs(out_dir, exist_ok=True)


def b64(arr):
    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(np.uint32(len(raw)).tobytes() + raw).decode()


def write_vtp(path, pts, species):
    n = len(pts)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(n, dtype=pts.dtype)])
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n')
        f.write(f'<PolyData><Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('<PointData Scalars="species">\n')
        f.write(f'<DataArray type="Int8" Name="species" format="binary">{b64(species.astype(np.int8))}</DataArray>\n')
        f.write('</PointData>\n<Points>\n')
        f.write(f'<DataArray type="Float32" NumberOfComponents="3" format="binary">{b64(pts.astype(np.float32))}</DataArray>\n')
        f.write('</Points>\n<Verts>\n')
        f.write(f'<DataArray type="Int32" Name="connectivity" format="binary">{b64(np.arange(n, dtype=np.int32))}</DataArray>\n')
        f.write(f'<DataArray type="Int32" Name="offsets" format="binary">{b64(np.arange(1, n + 1, dtype=np.int32))}</DataArray>\n')
        f.write('</Verts>\n</Piece></PolyData>\n</VTKFile>\n')


entries = []
for f in files:
    d = np.load(f)
    pts = np.concatenate([d["pos"], d["neg"]])
    species = np.concatenate([np.ones(len(d["pos"])), -np.ones(len(d["neg"]))])
    name = os.path.basename(f).replace("pts_", "frame_").replace(".npz", ".vtp")
    write_vtp(os.path.join(out_dir, name), pts, species)
    entries.append((float(d["t"]), name))

with open(os.path.join(out_dir, "simulation.pvd"), "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, name in entries:
        f.write(f'  <DataSet timestep="{t}" file="{name}"/>\n')
    f.write('</Collection>\n</VTKFile>\n')

print(f"wrote {len(entries)} frames to {out_dir} (open simulation.pvd in ParaView)")