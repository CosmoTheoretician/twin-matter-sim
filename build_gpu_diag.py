import json
import re
import shutil
from pathlib import Path
import anthropic

SRC = Path("sim_gpu")
DST = Path("sim_gpu2")
PLAN_FILE = Path("gpu_diag_plan.md")
MODEL = "claude-fable-5-1"

client = anthropic.Anthropic()

GOAL = """This N-body simulation already runs its forces and integrator on an AMD GPU via PyTorch
(ROCm, "cuda" device name), validated against the numpy CPU path. But in a 6.4 million particle
run, forces took 178 s of a 1604 s run: about 89% of the time is host-side work at diagnostic,
structure and snapshot cadences (copying particles to the host, energy and momentum sums,
density grids, clump statistics, segregation, persistence, power spectra, and plotting).

Move as much of this as possible onto the GPU, so large 2-D and 3-D runs are limited by the
simulation itself. Requirements:

1. Compute energy, momentum, velocity dispersions and similar sums on the device.
2. Build density grids on the device (deposition with index_add), and compute grid-level
   statistics there (segregation, persistence, power spectra and peak wavelength). Only copy
   small grids to the host where truly needed, for example connected-component clump labelling
   with scipy.ndimage on a structure grid, which is cheap at grid sizes up to 512 squared.
3. Snapshots: compute density maps on the device and copy only the maps. For particle scatter
   panels, copy a fixed random subsample (default 200000 particles per population, settable
   with a command-line option) instead of all particles, and state the subsample in the panel.
4. Never copy the full particle arrays to the host during a run, except for an optional final
   state dump.
5. The CPU (--device cpu) path must stay unchanged. The numerical definitions of every metric
   must stay identical, so results remain comparable with earlier runs.
6. Add a timing breakdown printed at the end of every run: forces, integration, energy
   diagnostics, structure statistics, snapshots and other, in seconds and percent.
7. Add a "diagtest" command that builds one particle state, computes every diagnostic and
   structure metric on the CPU path and the GPU path, and reports the differences, so the port
   can be validated (float64 on the GPU must agree to near machine precision).
8. Each file must fit in one response (at most about 500 lines). Split large changes into new
   modules rather than growing existing files, and change as few files as possible."""

PLAN_PROMPT = GOAL + """

Below is the complete current code. Write:
1. A short Markdown "Plan" section: which files change, which are new, and what moves to the GPU.
2. A ```json code block listing every file to write, in dependency order, each with "file",
   "action" ("new" or "replace") and "changes" (what must be in that file).

Do not write any code yet; the files will be requested one at a time afterwards.

=== CURRENT CODE ===
CODE_PLACEHOLDER"""

FILE_PROMPT = GOAL + """

You are now writing one file of this change, following the plan below exactly.

File: FILE_PLACEHOLDER (ACTION_PLACEHOLDER)
Required changes: CHANGES_PLACEHOLDER

Respond with only a single ```python code block containing the complete file.

=== PLAN ===
PLAN_PLACEHOLDER

=== CURRENT CODE (including files already written for this change) ===
CODE_PLACEHOLDER"""


def all_code(folder):
    return "\n\n".join(f"# ===== {p.name} =====\n{p.read_text(encoding='utf-8')}"
                       for p in sorted(folder.glob("*.py")))


def ask(prompt, label):
    print(f"{label} (this can take several minutes)...")
    with client.messages.stream(model=MODEL, max_tokens=64000,
                                messages=[{"role": "user", "content": prompt}]) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                print(".", end="", flush=True)
        msg = stream.get_final_message()
    print()
    if msg.stop_reason == "max_tokens":
        print("Warning: response was cut off.")
    return "".join(b.text for b in msg.content if b.type == "text")


def main():
    if not DST.exists():
        DST.mkdir()
        for p in list(SRC.glob("*.py")) + list(SRC.glob("*.sh")):
            shutil.copy(p, DST / p.name)
        print(f"Copied {SRC}/ code to {DST}/")

    if PLAN_FILE.exists() and "```json" in PLAN_FILE.read_text(encoding="utf-8"):
        plan = PLAN_FILE.read_text(encoding="utf-8")
        print("Using existing plan")
    else:
        plan = ask(PLAN_PROMPT.replace("CODE_PLACEHOLDER", all_code(SRC)),
                   "Step 1: diagnostics port plan")
        PLAN_FILE.write_text(plan, encoding="utf-8")

    files = json.loads(re.search(r"```json\n(.*?)```", plan, re.DOTALL).group(1))
    print(f"Files to write: {[f['file'] for f in files]}")

    done_marker = DST / ".done_diag"
    done = set(done_marker.read_text().split()) if done_marker.exists() else set()

    for i, f in enumerate(files, 1):
        if f["file"] in done:
            print(f"Skipping {f['file']} (already written)")
            continue
        changes = f["changes"] if isinstance(f["changes"], str) else json.dumps(f["changes"])
        prompt = (FILE_PROMPT.replace("FILE_PLACEHOLDER", f["file"])
                  .replace("ACTION_PLACEHOLDER", f["action"])
                  .replace("CHANGES_PLACEHOLDER", changes)
                  .replace("PLAN_PLACEHOLDER", plan)
                  .replace("CODE_PLACEHOLDER", all_code(DST)))
        text = ask(prompt, f"Step 2.{i}/{len(files)}: {f['file']}")
        m = re.search(r"```python\n(.*?)(```|\Z)", text, re.DOTALL)
        if m:
            (DST / f["file"]).write_text(m.group(1), encoding="utf-8")
            done.add(f["file"])
            done_marker.write_text("\n".join(sorted(done)))
            print(f"Wrote {DST / f['file']}")
        else:
            (DST / f"{f['file']}.raw.txt").write_text(text, encoding="utf-8")
            print(f"No code found for {f['file']}; raw response saved")

    print(f"Done. Test with: cd {DST} && python main.py diagtest")


if __name__ == "__main__":
    main()
