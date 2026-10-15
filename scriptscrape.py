import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup
import anthropic

START_URL = "https://www.jp-petit.org/"
MAX_PAGES = 50000
DELAY = 1.0  # seconds between requests
OUT_DIR = Path("pages")
RESULTS_FILE = Path("analysis.jsonl")

ANALYSIS_PROMPT = """Analyze this web page. Summarize its purpose and main content,
and note anything notable. Also point out what category broadly it goes into and mark science content.s

URL: {url}

Page text:
{text}"""


# ---------- Stage 1: crawl and save ----------

def crawl():
    OUT_DIR.mkdir(exist_ok=True)
    domain = urlparse(START_URL).netloc
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (site-analysis crawler)"

    queue = [START_URL]
    seen = {START_URL}
    count = 0

    while queue and count < MAX_PAGES:
        url = queue.pop(0)
        try:
            resp = session.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"ERROR {url}: {e}")
            continue

        if resp.status_code != 200 or "text/html" not in resp.headers.get("Content-Type", ""):
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # queue internal links
        for a in soup.find_all("a", href=True):
            link, _ = urldefrag(urljoin(url, a["href"]))
            if urlparse(link).netloc == domain and link not in seen:
                seen.add(link)
                queue.append(link)

        # extract readable text
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.get_text(strip=True) if soup.title else ""

        name = hashlib.md5(url.encode()).hexdigest()
        (OUT_DIR / f"{name}.html").write_text(resp.text, encoding="utf-8")
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )

        count += 1
        print(f"[{count}] {url}")
        time.sleep(DELAY)


# ---------- Stage 2: analyze each page ----------

def analyze():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    done = set()
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["url"])

    with RESULTS_FILE.open("a", encoding="utf-8") as out:
        for f in sorted(OUT_DIR.glob("*.json")):
            page = json.loads(f.read_text(encoding="utf-8"))
            if page["url"] in done:
                continue

            text = page["text"][:50_000]  # keep very long pages in bounds
            msg = client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(url=page["url"], text=text)}],
            )
            analysis = "".join(b.text for b in msg.content if b.type == "text")
            result = {"url": page["url"], "title": page["title"], "analysis": analysis}
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Analyzed {page['url']}")


if __name__ == "__main__":
    #crawl()
    analyze()