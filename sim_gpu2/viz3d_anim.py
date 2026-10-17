import sys
import glob
import numpy as np
import plotly.graph_objects as go

run = sys.argv[1].rstrip("/")
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
files = sorted(glob.glob(f"{run}/pts_*.npz"))
if not files:
    sys.exit("No pts_*.npz files found; was the run made after the patch?")

frames_data, times = [], []
for f in files:
    d = np.load(f)
    frames_data.append((d["pos"][:n_show], d["neg"][:n_show]))
    times.append(float(d["t"]))
allp = np.concatenate([np.concatenate([p, n]) for p, n in frames_data if len(p) + len(n)])
lo, hi = allp.min(axis=0), allp.max(axis=0)

def traces(p, n):
    return [go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="ordinary matter",
                         marker=dict(size=1.2, color="#3b82f6", opacity=0.35)),
            go.Scatter3d(x=n[:, 0], y=n[:, 1], z=n[:, 2], mode="markers", name="twin matter",
                         marker=dict(size=1.2, color="#ef4444", opacity=0.35))]

fig = go.Figure(data=traces(*frames_data[0]),
                frames=[go.Frame(data=traces(p, n), name=str(i)) for i, (p, n) in enumerate(frames_data)])
fig.update_layout(
    title=f"{run.split('/')[-1]}  ({len(files)} snapshots, {n_show:,} particles per population shown)",
    uirevision="keep", paper_bgcolor="black", font_color="white", legend=dict(itemsizing="constant"),
    scene=dict(aspectmode="cube", bgcolor="black",
               xaxis=dict(range=[lo[0], hi[0]]), yaxis=dict(range=[lo[1], hi[1]]), zaxis=dict(range=[lo[2], hi[2]])),
    updatemenus=[dict(type="buttons", x=0.05, y=0.05, buttons=[
        dict(label="Play", method="animate",
             args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
        dict(label="Pause", method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    sliders=[dict(x=0.15, len=0.8, currentvalue=dict(prefix="t = "), steps=[
        dict(label=f"{tt:.1f}", method="animate",
             args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
        for i, tt in enumerate(times)])])
out = "/mnt/c/Users/roche/Desktop/" + run.split("/")[-1] + "_3d_anim.html"
fig.write_html(out)
print("saved", out)