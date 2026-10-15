import json
from pathlib import Path
import anthropic

RESULTS_FILE = Path("analysis.jsonl")
BATCH_DIR = Path("batches")
FINAL_FILE = Path("final_report.md")
MODEL = "claude-opus-5-5"
BATCH_CHARS = 150_000  # size of each batch of summaries

client = anthropic.Anthropic()

BATCH_PROMPT = """Below are AI-written summaries of pages from one website, each with its URL.

For this batch:
1. List the main themes/categories the pages fall into, with the URLs belonging to each.
2. Identify every page with scientific content (physics, biology, astronomy, etc.) and categorize it by scientific field, with URLs.
3. Pick out the most interesting, surprising or noteworthy pages and explain briefly why, with URLs.

Summaries:
{summaries}"""

FINAL_PROMPT = """Below are partial analyses, each covering one batch of pages from the same website.
Merge them into one final report in Markdown with these sections:

1. General summary: what the site is about overall, its main purposes and perspective.
2. Categories: a unified, deduplicated set of categories with a short description and key URLs for each.
3. Science content: all scientific material organized by field, with a short description and URLs.
4. Highlights: the most interesting or noteworthy pages across the whole site and why.

Partial analyses:
{batches}"""


def ask(prompt):
    with client.messages.stream(
        model=MODEL,
        max_tokens=25000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    return "".join(b.text for b in msg.content if b.type == "text")


def make_batches():
    entries = []
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("analysis"):
            entries.append(f"URL: {r['url']}\nTitle: {r['title']}\n{r['analysis']}\n")

    batches, current, size = [], [], 0
    for e in entries:
        if size + len(e) > BATCH_CHARS and current:
            batches.append(current)
            current, size = [], 0
        current.append(e)
        size += len(e)
    if current:
        batches.append(current)
    return batches


def main():
    BATCH_DIR.mkdir(exist_ok=True)
    batches = make_batches()
    print(f"{len(batches)} batches")

    for i, batch in enumerate(batches):
        out = BATCH_DIR / f"batch_{i:03d}.md"
        if out.exists():
            continue
        try:
            out.write_text(ask(BATCH_PROMPT.format(summaries="\n---\n".join(batch))), encoding="utf-8")
            print(f"Batch {i + 1}/{len(batches)} done")
        except anthropic.APIStatusError as e:
            print(f"SKIPPED batch {i}: {e.message}")

    combined = "\n\n=====\n\n".join(p.read_text(encoding="utf-8") for p in sorted(BATCH_DIR.glob("batch_*.md")))
    FINAL_FILE.write_text(ask(FINAL_PROMPT.format(batches=combined)), encoding="utf-8")
    print(f"Final report written to {FINAL_FILE}")


if __name__ == "__main__":
    main()