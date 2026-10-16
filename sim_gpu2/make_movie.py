import sys
from pathlib import Path
from PIL import Image

folder = Path(sys.argv[1])
pattern = sys.argv[2] if len(sys.argv) > 2 else "*snap*.png"
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 4

files = sorted(folder.rglob(pattern))
if not files:
    print(f"No files matching {pattern} in {folder}")
    sys.exit(1)

print(f"{len(files)} frames: {files[0].name} ... {files[-1].name}")
frames = [Image.open(f).convert("RGB") for f in files]
out = folder / "movie.gif"
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=int(1000 / fps), loop=0)
print(f"Saved {out}")