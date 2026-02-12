"""
Treasury Select Committee — oral evidence transcript extractor.

Pulls every downloadable oral-evidence session from the UK Parliament
Committees API (Committee ID 158), decodes the base64-encoded HTML
transcript, strips it to plain text, and saves:

    tsc_data/transcripts/{evidence_id}.txt   – one file per session
    tsc_data/metadata.csv                    – date, title, witnesses, …

No authentication required.
"""

import requests
import time
import csv
import base64
import re
from html.parser import HTMLParser
from pathlib import Path

API = "https://committees-api.parliament.uk/api"
COMMITTEE_ID = 158
PAGE_SIZE = 50
OUT_DIR = Path(__file__).parent / "tsc_data"
TRANSCRIPT_DIR = OUT_DIR / "transcripts"

# ── helpers ──────────────────────────────────────────────────────────

class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML → plain-text converter."""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str):
        self._parts.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        # collapse whitespace
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _get_json(url: str, params: dict | None = None,
              retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED {url}: {e}", flush=True)
                return None


def _decode_html_transcript(evidence_id: int) -> str | None:
    """Download + decode a single oral-evidence HTML transcript."""
    data = _get_json(f"{API}/OralEvidence/{evidence_id}/Document/Html")
    if not data or "data" not in data:
        return None
    try:
        html = base64.b64decode(data["data"]).decode("utf-8")
    except Exception:
        return None
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── main ─────────────────────────────────────────────────────────────

def main():
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Paginate through all oral-evidence items
    print("Fetching oral-evidence index …", flush=True)
    items: list[dict] = []
    skip = 0
    while True:
        page = _get_json(f"{API}/OralEvidence",
                         params={"CommitteeId": COMMITTEE_ID,
                                 "Take": PAGE_SIZE, "Skip": skip})
        if not page or not page.get("items"):
            break
        items.extend(page["items"])
        total = page.get("totalResults", "?")
        print(f"  fetched {len(items)}/{total}", flush=True)
        skip += PAGE_SIZE
        time.sleep(0.3)

    print(f"\n{len(items)} oral-evidence sessions found.\n", flush=True)

    # 2. Download each transcript
    rows: list[dict] = []
    success = skipped = failed = 0

    for i, item in enumerate(items, 1):
        eid = item["id"]
        date = (item.get("meetingDate") or "")[:10]
        businesses = item.get("committeeBusinesses") or []
        title = businesses[0]["title"] if businesses else ""
        biz_id = businesses[0]["id"] if businesses else ""
        witnesses = "; ".join(
            w.get("name") or "" for w in (item.get("witnesses") or [])
        )
        has_doc = item.get("document") is not None

        txt_path = TRANSCRIPT_DIR / f"{eid}.txt"

        # skip if already downloaded
        if txt_path.exists() and txt_path.stat().st_size > 0:
            skipped += 1
            rows.append(dict(evidence_id=eid, date=date, inquiry=title,
                             inquiry_id=biz_id, witnesses=witnesses,
                             chars=txt_path.stat().st_size, file=txt_path.name))
            print(f"  [{i}/{len(items)}] {date}  {title[:50]}  — cached",
                  flush=True)
            continue

        if not has_doc:
            failed += 1
            print(f"  [{i}/{len(items)}] {date}  {title[:50]}  — no document",
                  flush=True)
            continue

        text = _decode_html_transcript(eid)
        if not text:
            failed += 1
            print(f"  [{i}/{len(items)}] {date}  {title[:50]}  — decode failed",
                  flush=True)
            continue

        txt_path.write_text(text, encoding="utf-8")
        success += 1
        rows.append(dict(evidence_id=eid, date=date, inquiry=title,
                         inquiry_id=biz_id, witnesses=witnesses,
                         chars=len(text), file=txt_path.name))
        print(f"  [{i}/{len(items)}] {date}  {title[:50]}  "
              f"({len(text):,} chars)", flush=True)

        time.sleep(0.4)  # be polite

    # 3. Write metadata CSV
    rows.sort(key=lambda r: r["date"])
    meta_path = OUT_DIR / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone.  success={success}  cached={skipped}  failed={failed}")
    print(f"Transcripts -> {TRANSCRIPT_DIR}")
    print(f"Metadata    -> {meta_path}")


if __name__ == "__main__":
    main()
