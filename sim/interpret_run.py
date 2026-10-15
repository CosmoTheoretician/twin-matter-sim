import sys
import base64
from pathlib import Path
import anthropic

MODEL = "claude-fable-5-1"
INVESTIGATION = Path("../astrophysics_investigation.md")
MAX_IMAGES = 20
MAX_TEXT_CHARS = 30_000

PROMPT = """You are reviewing the output of a 2-D N-body simulation that tests the "twin matter"
claims described in the investigation below. Negative-mass (twin) particles attract each other,
positive-mass particles attract each other, and the two populations repel each other.

You are given snapshot images in time order (filenames indicate the time), plus the run's
measured data. Interpret the run neutrally and rigorously:

1. Describe what the images actually show over time: when structure appears, how it evolves,
   where each population ends up.
2. Check the images against the measured data and point out any agreement or mismatch.
3. Compare the outcome with what the original claims predict (cellular pattern, stable clumps,
   long-lived structure), clearly separating what the run supports, contradicts, or cannot decide.
4. Point out any visual signs of numerical artefacts (grid patterns, box-size effects, noise,
   periodic-boundary effects).
5. Suggest the most informative next runs.

Neither advocate for nor dismiss the original claims. State your uncertainty honestly, and note
that this is a single run unless the data says otherwise.

=== INVESTIGATION (context) ===
{investigation}

=== RUN DATA ===
{data}"""


def read_text_files(folder):
    parts = []
    for pattern in ["summary*.json", "assumptions*.txt", "structure*.csv"]:
        for f in sorted(folder.rglob(pattern)):
            parts.append(f"--- {f.name} ---\n{f.read_text(encoding='utf-8')[:MAX_TEXT_CHARS]}")
    return "\n\n".join(parts) or "(no data files found)"


def pick_images(folder, pattern):
    files = sorted(folder.rglob(pattern))
    if len(files) > MAX_IMAGES:
        step = (len(files) - 1) / (MAX_IMAGES - 1)
        files = [files[round(i * step)] for i in range(MAX_IMAGES)]
    return files


def main():
    if len(sys.argv) < 2:
        print("Usage: python interpret_run.py <run_folder> [image_pattern]")
        return
    folder = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "*snap*.png"

    images = pick_images(folder, pattern)
    if not images:
        print(f"No images matching {pattern} in {folder}")
        return
    print(f"Sending {len(images)} images: {images[0].name} ... {images[-1].name}")

    investigation = INVESTIGATION.read_text(encoding="utf-8") if INVESTIGATION.exists() else "(not found)"
    content = []
    for f in images:
        content.append({"type": "text", "text": f"Image: {f.name}"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.standard_b64encode(f.read_bytes()).decode()}})
    content.append({"type": "text", "text": PROMPT.format(
        investigation=investigation, data=read_text_files(folder))})

    client = anthropic.Anthropic()
    print("Sending to Fable (this can take several minutes)...")
    with client.messages.stream(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                print(".", end="", flush=True)
        msg = stream.get_final_message()

    out = folder / "interpretation.md"
    out.write_text("".join(b.text for b in msg.content if b.type == "text"), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()