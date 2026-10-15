import json
import re
from pathlib import Path
import anthropic

INVESTIGATION_FILE = Path("astrophysics_investigation.md")
PAGES_DIR = Path("pages")
NOTES_FILE = Path("simulation_notes.md")
SIM_DIR = Path("sim")
MODEL = "claude-fable-5-1"
MAX_PAGES = 25
MAX_PAGE_CHARS = 20_000
MAX_TOTAL_CHARS = 300_000

KEYWORDS = [
    "twin matter", "matière gémellaire", "matiere gemellaire", "gémellaire", "jumelle",
    "n-body", "n-corps", "n corps", "simulation", "spiral", "spirale",
    "galaxie", "galaxy", "masse négative", "negative mass", "janus",
]

client = anthropic.Anthropic()

PROMPT = """Below is a neutral investigation of the "twin matter" N-body simulations on a website,
followed by the original pages describing them.

Design a Python simulation that lets a user test how far these results can be reproduced and
where they break down. Requirements:

1. Faithful reproduction first: implement the interaction laws, dimensionality, particle counts,
   initial conditions and boundary conditions as described in the source pages. Where the pages
   are vague or silent, choose a reasonable default, and list every such assumption.
2. Two scenarios: (a) large-scale structure (two populations, periodic box, looking for the
   cellular pattern), and (b) the spiral galaxy (a disk inside a surrounding population).
3. Stress tests to find where results break down, selectable from the command line: increasing
   particle count, varying softening and timestep, longer run times, the population mass/density
   ratio, and if feasible a 3-D mode and cosmic expansion.
4. Diagnostics: energy and momentum conservation, and simple quantitative measures (e.g. a
   clustering/cell statistic for LSS, and bar/spiral strength over time for the disk), saved to CSV.
5. Output: PNG snapshots and plots saved to an output folder. No interactive windows.
6. Must run on an ordinary PC under Linux using only numpy, scipy and matplotlib. Use an efficient
   force method (e.g. FFT particle-mesh for the periodic box) so higher particle counts are practical.
7. Be scientifically honest: do not tune parameters to force the claimed result. Report negative
   or ambiguous outcomes plainly.

=== INVESTIGATION ===
{investigation}

=== SOURCE PAGES ===
{pages}"""

NOTES_INSTRUCTION = """

For this response, write ONLY the design, not the code. The code will be requested afterwards
one file at a time, and each response has a strict output limit, so the code MUST be split into
modules where every file fits easily in one response (at most about 500 lines per file).

Write:
1. A "Notes" section in Markdown: assumptions, overall design, command-line options, example
   commands, and what outcomes would support or undermine the original claims.
2. A module plan as a single ```json code block: a list of objects, in the order they should be
   written (dependencies first), each with "file" (e.g. "forces.py"), "purpose", and "interface"
   (the exact functions/classes it exposes, with signatures, and which modules it imports from).
   Include a "main.py" entry point last."""

MODULE_PROMPT = """You are writing one file of a modular twin-matter N-body simulation.
Each response has a strict output limit, so write ONLY the file requested, keep it compact
(at most about 500 lines), and follow the module plan's interfaces exactly so the files fit together.

File to write now: {file}
Purpose: {purpose}
Interface: {interface}

Respond with only a single ```python code block containing the complete file.

=== DESIGN NOTES AND MODULE PLAN ===
{notes}

=== FILES ALREADY WRITTEN ===
{written}

=== INVESTIGATION ===
{investigation}

=== SOURCE PAGES ===
{pages}"""


def find_pages():
    scored = []
    for f in PAGES_DIR.glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        hay = (p["title"] + " " + p["text"]).lower()
        hits = {k for k in KEYWORDS if k in hay}
        if len(hits) >= 2:
            scored.append((sum(hay.count(k) for k in hits), p))
    scored.sort(key=lambda x: x[0], reverse=True)

    chunks, total = [], 0
    for score, p in scored[:MAX_PAGES]:
        chunk = f"URL: {p['url']}\nTitle: {p['title']}\n{p['text'][:MAX_PAGE_CHARS]}\n"
        if total + len(chunk) > MAX_TOTAL_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
    return chunks


def ask(prompt, label):
    print(f"{label} (this can take several minutes)...")
    with client.messages.stream(
        model=MODEL,
        max_tokens=64000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                print(".", end="", flush=True)
        msg = stream.get_final_message()
    print()
    if msg.stop_reason == "max_tokens":
        print("Warning: response was cut off.")
    return "".join(b.text for b in msg.content if b.type == "text")


def main():
    investigation = INVESTIGATION_FILE.read_text(encoding="utf-8")
    pages = "\n---\n".join(find_pages())
    SIM_DIR.mkdir(exist_ok=True)

    if NOTES_FILE.exists() and "```json" in NOTES_FILE.read_text(encoding="utf-8"):
        notes = NOTES_FILE.read_text(encoding="utf-8")
        print("Using existing notes and module plan")
    else:
        notes = ask(PROMPT.format(investigation=investigation, pages=pages) + NOTES_INSTRUCTION,
                    "Step 1: design notes and module plan")
        NOTES_FILE.write_text(notes, encoding="utf-8")
        print(f"Notes written to {NOTES_FILE}")

    plan_match = re.search(r"```json\n(.*?)```", notes, re.DOTALL)
    if not plan_match:
        print("No module plan found in the notes; delete simulation_notes.md and rerun.")
        return
    plan = json.loads(plan_match.group(1))
    print(f"Module plan: {[m['file'] for m in plan]}")

    for i, module in enumerate(plan, 1):
        path = SIM_DIR / module["file"]
        if path.exists():
            print(f"Skipping {module['file']} (already written)")
            continue

        written = "\n\n".join(
            f"# ===== {p.name} =====\n{p.read_text(encoding='utf-8')}"
            for p in sorted(SIM_DIR.glob("*.py"))
        ) or "(none yet)"

        interface = module["interface"]
        if not isinstance(interface, str):
            interface = json.dumps(interface)

        text = ask(MODULE_PROMPT.format(
            file=module["file"], purpose=module["purpose"], interface=interface,
            notes=notes, written=written, investigation=investigation, pages=pages),
            f"Step 2.{i}/{len(plan)}: {module['file']}")

        match = re.search(r"```python\n(.*?)(```|\Z)", text, re.DOTALL)
        if match:
            path.write_text(match.group(1), encoding="utf-8")
            print(f"Wrote {path}")
        else:
            (SIM_DIR / f"{module['file']}.raw.txt").write_text(text, encoding="utf-8")
            print(f"No code found for {module['file']}; raw response saved")

    print("Done. Run the simulation from the sim folder, e.g.: cd sim && python main.py --help")


if __name__ == "__main__":
    main()