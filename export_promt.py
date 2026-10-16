import json
import re
from pathlib import Path
import build_gpu_diag as b

plan = b.PLAN_FILE.read_text(encoding="utf-8")
files = json.loads(re.search(r"```json\n(.*?)```", plan, re.DOTALL).group(1))
marker = b.DST / ".done_diag"
done = set(marker.read_text().split()) if marker.exists() else set()

for f in files:
    if f["file"] in done:
        continue
    changes = f["changes"] if isinstance(f["changes"], str) else json.dumps(f["changes"])
    prompt = (b.FILE_PROMPT.replace("FILE_PLACEHOLDER", f["file"])
              .replace("ACTION_PLACEHOLDER", f["action"])
              .replace("CHANGES_PLACEHOLDER", changes)
              .replace("PLAN_PLACEHOLDER", plan)
              .replace("CODE_PLACEHOLDER", b.all_code(b.DST)))
    out = Path(f"/mnt/c/Users/roche/Desktop/prompt_{f['file']}.txt")
    out.write_text(prompt, encoding="utf-8")
    print(f"Next file: {f['file']} -> prompt saved to {out}")
    break
else:
    print("All files are done.")