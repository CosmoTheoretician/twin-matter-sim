import json
import re
from pathlib import Path
import anthropic

REPORT_FILE = Path("final_report.md")
PAGES_DIR = Path("pages")
OUT_FILE = Path("astrophysics_investigation.md")
MODEL = "claude-fable-5-1"
TOPIC = "computational astrophysics"
FOCUS = 'Focus on the N-body simulations of "twin matter" and spiral galaxies.'
MAX_PAGES = 40
MAX_PAGE_CHARS = 20_000
MAX_TOTAL_CHARS = 400_000

KEYWORDS = [
    "twin matter", "matière gémellaire", "matiere gemellaire", "gémellaire", "jumelle",
    "n-body", "n-corps", "n corps", "simulation", "spiral", "spirale",
    "galaxie", "galaxy", "masse négative", "negative mass", "janus",
]

client = anthropic.Anthropic()

PROMPT = """You are investigating the {topic} work presented on a website. Below you have
(1) an overall report about the site, and (2) the full text of the relevant pages.

{focus} Use other pages only as context where they help explain this work.

Investigate this work neutrally and rigorously. Neither advocate for it nor dismiss it.
In Markdown, cover:

1. Overview: what the work claims and proposes, in the author's own terms.
2. Methods: the computational and theoretical methods used, and how they are described.
3. Relation to mainstream science: where it agrees with, extends, or departs from established
   astrophysics and cosmology, and how it has been received (peer review, publications, critiques),
   noting clearly when you are unsure.
4. Strengths: what is well-founded, original or worth taking seriously.
5. Weaknesses and open questions: gaps, unsupported claims, testability issues, or errors.
6. Conclusion: a balanced assessment, clearly separating what is established, what is plausible
   but unproven, and what is not supported.

Always distinguish the author's claims from verified facts, cite the page URLs you rely on,
and state your uncertainty honestly.

=== SITE REPORT ===
{report}

=== RELEVANT PAGES ===
{pages}"""


def find_pages():
    scored = []
    for f in PAGES_DIR.glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        hay = (p["title"] + " " + p["text"]).lower()
        hits = {k for k in KEYWORDS if k in hay}
        score = sum(hay.count(k) for k in hits)
        if len(hits) >= 2:  # require at least two different keywords
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)

    chunks, total = [], 0
    for score, p in scored[:MAX_PAGES]:
        chunk = f"URL: {p['url']}\nTitle: {p['title']}\n{p['text'][:MAX_PAGE_CHARS]}\n"
        if total + len(chunk) > MAX_TOTAL_CHARS:
            break
        chunks.append(chunk)
        total += len(chunk)
        print(f"  [{score}] {p['url']}")
    return chunks


def main():
    report = REPORT_FILE.read_text(encoding="utf-8")
    pages = find_pages()
    print(f"Loaded {len(pages)} pages")
    if not pages:
        print("No matching pages found, stopping.")
        return

    print("Sending to Fable (this can take several minutes)...")
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        messages=[{"role": "user", "content": PROMPT.format(
            topic=TOPIC, focus=FOCUS, report=report, pages="\n---\n".join(pages))}],
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                print(".", end="", flush=True)
        msg = stream.get_final_message()

    OUT_FILE.write_text("".join(b.text for b in msg.content if b.type == "text"), encoding="utf-8")
    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()