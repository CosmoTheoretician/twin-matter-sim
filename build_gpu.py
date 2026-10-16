import json
import re
import shutil
from pathlib import Path
import anthropic

SRC = Path("sim")
DST = Path("sim_gpu")
NOTES = Path("simulation_notes.md")
PLAN_FILE = Path("gpu_plan.md")
MODEL = "claude-fable-5-1"

client = anthropic.Anthropic()

GOAL = """Port this N-body simulation to run on an AMD Radeon RX 9060 XT (16 GB) GPU via PyTorch
with ROCm under WSL (PyTorch exposes AMD GPUs through the "cuda" device name). Requirements:

1. Add a --device {{cpu,gpu}} option. --device cpu must keep the existing numpy code path unchanged,
   so it remains the validated reference.
2. On the GPU, keep particle positions and velocities resident on the device for the whole run.
   Copy to the host only at diagnostic, structure and snapshot cadences.
3. Port the particle-mesh force calculation for both 2-D and 3-D: mass deposition (scatter/index_add),
   FFT Poisson solve with torch.fft, force computation, and interpolation back to particles,
   using exactly the same scheme as the CPU code so results are directly comparable.
4. Port the leapfrog integrator, including the expansion option, to run on the device.
5. Default to float32 on the GPU (consumer GPUs are slow at float64), but allow --dtype float64
   on the GPU for validation.
6. Add a "gputest" command that compares GPU and CPU forces on the same random particles in 2-D
   and 3-D and reports relative errors, and prints GPU memory use and time per force evaluation.
7. Print an estimate of GPU memory needed before a run starts, and stop with a clear message if it
   will not fit in 16 GB.
8. Make sure 3-D snapshots are informative: projected density maps and a thin slice for each
   population.
9. Change as few files as possible. Each file must fit in one response (at most about 500 lines),
   so split large changes into new modules rather than growing existing files."""

PLAN_PROMPT = GOAL + """

Below are the design notes and the complete current code. Write:
1. A short Markdown "Plan" section: which files change, which are new, and how the GPU path works.
2. A ```json code block listing every file to write, in dependency order, each with "file",
   "action" ("new" or "replace") and "changes" (what must be in that file).

Do not write any code yet; the files will be requested one at a time afterwards.

=== DESIGN NOTES ===
{notes}

=== CURRENT CODE ===
{code}"""

FILE_PROMPT = GOAL + """

You are now writing one file of the GPU port, following the plan below exactly.

File: {file} ({action})
Required changes: {changes}

Respond with only a single ```python code block containing the complete file.

=== PLAN ===
{plan}

=== CURRENT CODE (including files already written for the port) ===
{code}"""


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
        for p in SRC.glob("*.py"):
            shutil.copy(p, DST / p.name)
        for p in SRC.glob("*.sh"):
            shutil.copy(p, DST / p.name)
        print(f"Copied simulation code to {DST}/")

    if PLAN_FILE.exists() and "```json" in PLAN_FILE.read_text(encoding="utf-8"):
        plan = PLAN_FILE.read_text(encoding="utf-8")
        print("Using existing plan")
    else:
        notes = NOTES.read_text(encoding="utf-8") if NOTES.exists() else "(none)"
        plan = ask(PLAN_PROMPT.format(notes=notes, code=all_code(SRC)), "Step 1: GPU port plan")
        PLAN_FILE.write_text(plan, encoding="utf-8")

    files = json.loads(re.search(r"```json\n(.*?)```", plan, re.DOTALL).group(1))
    print(f"Files to write: {[f['file'] for f in files]}")

    done_marker = DST / ".done"
    done = set(done_marker.read_text().split()) if done_marker.exists() else set()

    for i, f in enumerate(files, 1):
        if f["file"] in done:
            print(f"Skipping {f['file']} (already written)")
            continue
        changes = f["changes"] if isinstance(f["changes"], str) else json.dumps(f["changes"])
        text = ask(FILE_PROMPT.format(file=f["file"], action=f["action"], changes=changes,
                                      plan=plan, code=all_code(DST)),
                   f"Step 2.{i}/{len(files)}: {f['file']}")
        m = re.search(r"```python\n(.*?)(```|\Z)", text, re.DOTALL)
        if m:
            (DST / f["file"]).write_text(m.group(1), encoding="utf-8")
            done.add(f["file"])
            done_marker.write_text("\n".join(sorted(done)))
            print(f"Wrote {DST / f['file']}")
        else:
            (DST / f"{f['file']}.raw.txt").write_text(text, encoding="utf-8")
            print(f"No code found for {f['file']}; raw response saved")

    print(f"Done. Test with: cd {DST} && python main.py gputest")


if __name__ == "__main__":
    main()
