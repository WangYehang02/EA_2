#!/usr/bin/env python
"""Download BSI QuakeML archives by scraping official archive pages (no invented URLs).

Supports resume + limited retries. Writes provenance (URL, DOI, sha256, timestamp).
Also supports offline import of already-downloaded ZIP files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DOIS = [
    "10.13127/BSI/202101",
    "10.13127/BSI/202102",
    "10.13127/BSI/202103",
    "10.13127/BSI/202201",
    "10.13127/BSI/202202",
    "10.13127/BSI/202203",
]

ARCHIVE_TMPL = "https://bsi.ingv.it/en/archivio-dati?doi={doi}"
USER_AGENT = "EA2-paper-strengthening/1.0 (research; contact: local)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, *, dest: Path | None = None, timeout: int = 120, retries: int = 5) -> bytes | None:
    """GET with cert verification ON. Bypass broken local HTTP proxies for INGV hosts."""
    import os

    import requests

    # Local MITM/proxy (e.g. 127.0.0.1:7890) causes TLS EOF to bsi.ingv.it; direct works.
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    proxies = {"http": None, "https": None}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if dest is None:
                r = requests.get(url, timeout=timeout, proxies=proxies, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
                return r.content
            tmp = dest.with_suffix(dest.suffix + ".part")
            # resume if .part exists
            headers = {"User-Agent": USER_AGENT}
            mode = "wb"
            pos = 0
            if tmp.exists():
                pos = tmp.stat().st_size
                if pos > 0:
                    headers["Range"] = f"bytes={pos}-"
                    mode = "ab"
            with requests.get(url, timeout=timeout, proxies=proxies, headers=headers, stream=True) as r:
                if r.status_code == 416:
                    # already complete?
                    break
                r.raise_for_status()
                with open(tmp, mode) as out:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            out.write(chunk)
            tmp.replace(dest)
            return None
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed {url} after {retries}: {type(last_err).__name__}: {last_err}")


def extract_qml_links(html: str, page_url: str) -> list[dict]:
    """Extract QuakeML zip hrefs from archive HTML."""
    # absolute or relative links ending with __QML.zip
    pat = re.compile(
        r'href=["\']([^"\']+__QML\.zip)["\']',
        re.IGNORECASE,
    )
    found = []
    for m in pat.finditer(html):
        href = m.group(1)
        if href.startswith("http"):
            url = href
        elif href.startswith("/"):
            url = "https://bsi.ingv.it" + href
        else:
            # relative like 2022/....zip under /files/bsi/
            if href.startswith("files/"):
                url = "https://bsi.ingv.it/" + href
            elif re.match(r"20\d{2}/", href):
                url = "https://bsi.ingv.it/files/bsi/" + href
            else:
                url = "https://bsi.ingv.it/files/bsi/" + href.lstrip("./")
        found.append({"href_raw": href, "url": url})
    # also catch bare filenames in text if no href (markdown conversion)
    if not found:
        for m in re.finditer(r"(20\d{2}/\d{8}_\d{8}__\d+__INGV__QML\.zip)", html):
            rel = m.group(1)
            found.append({"href_raw": rel, "url": "https://bsi.ingv.it/files/bsi/" + rel})
    # dedupe
    uniq = {}
    for x in found:
        uniq[x["url"]] = x
    return list(uniq.values())


def validate_zip(path: Path) -> dict:
    out = {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
    if not path.exists() or path.stat().st_size < 1000:
        out["ok"] = False
        out["reason"] = "missing_or_too_small"
        return out
    # reject HTML error pages saved as zip
    head = path.read_bytes()[:200].lower()
    if b"<html" in head or b"<!doctype" in head:
        out["ok"] = False
        out["reason"] = "looks_like_html_error_page"
        return out
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            out["n_members"] = len(names)
            out["sample_members"] = names[:5]
            # try read one xml-ish member
            xmlish = [n for n in names if n.lower().endswith((".xml", ".qml", ".quakeml")) or "qml" in n.lower()]
            out["xmlish_members"] = len(xmlish)
            if xmlish:
                data = zf.read(xmlish[0])[:500].lower()
                out["first_xml_has_quakeml"] = b"quakeml" in data or b"event" in data
            out["ok"] = True
            out["reason"] = "zip_ok"
    except zipfile.BadZipFile as e:
        out["ok"] = False
        out["reason"] = f"bad_zip:{e}"
    return out


def discover(outdir: Path) -> list[dict]:
    rows = []
    for doi in DOIS:
        page = ARCHIVE_TMPL.format(doi=doi)
        rec = {"doi": doi, "archive_page": page, "discovered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            html = http_get(page, timeout=60).decode("utf-8", errors="replace")
            (outdir / "pages").mkdir(parents=True, exist_ok=True)
            safe = doi.replace("/", "_")
            page_path = outdir / "pages" / f"{safe}.html"
            page_path.write_text(html, encoding="utf-8")
            links = extract_qml_links(html, page)
            rec["n_qml_links"] = len(links)
            rec["qml_links"] = links
            rec["discover_ok"] = len(links) >= 1
            if not links:
                rec["discover_error"] = "no_QML_zip_link_found_in_page"
        except Exception as e:
            rec["discover_ok"] = False
            rec["discover_error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
    return rows


def download_all(manifest: list[dict], raw_dir: Path) -> list[dict]:
    results = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in manifest:
        if not rec.get("discover_ok"):
            results.append({**rec, "download_ok": False, "skip_reason": "discover_failed"})
            continue
        for link in rec["qml_links"]:
            url = link["url"]
            fname = url.rstrip("/").split("/")[-1]
            dest = raw_dir / fname
            item = {
                "doi": rec["doi"],
                "archive_page": rec["archive_page"],
                "url": url,
                "dest": str(dest),
                "download_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            try:
                if dest.exists() and dest.stat().st_size > 1000:
                    item["resumed_or_cached"] = True
                else:
                    http_get(url, dest=dest, timeout=600, retries=5)
                    item["resumed_or_cached"] = False
                item["download_finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                item["sha256"] = sha256_file(dest)
                item["validation"] = validate_zip(dest)
                item["download_ok"] = bool(item["validation"].get("ok"))
            except Exception as e:
                item["download_ok"] = False
                item["error"] = f"{type(e).__name__}: {e}"
            results.append(item)
    return results


def offline_import(import_dir: Path, raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in sorted(import_dir.glob("*QML*.zip")):
        dest = raw_dir / src.name
        if not dest.exists() or sha256_file(dest) != sha256_file(src):
            dest.write_bytes(src.read_bytes())
        rows.append(
            {
                "source": str(src),
                "dest": str(dest),
                "sha256": sha256_file(dest),
                "validation": validate_zip(dest),
                "imported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": "offline_import",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/results/paper_strengthening_v1/bsi_catalog"))
    ap.add_argument("--import-dir", type=Path, default=None, help="offline import directory of QML zips")
    ap.add_argument("--discover-only", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    if args.import_dir:
        rows = offline_import(args.import_dir, raw)
        (out / "offline_import_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({"imported": len(rows), "ok": sum(1 for r in rows if r["validation"].get("ok"))}, indent=2))
        return 0 if all(r["validation"].get("ok") for r in rows) else 2

    manifest = discover(out)
    (out / "discovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.discover_only:
        print(json.dumps(manifest, indent=2))
        return 0 if all(r.get("discover_ok") for r in manifest) else 2

    results = download_all(manifest, raw)
    (out / "download_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = sum(1 for r in results if r.get("download_ok"))
    print(json.dumps({"n": len(results), "ok": ok}, indent=2))
    for r in results:
        print(r.get("doi"), r.get("download_ok"), r.get("url"), r.get("error") or r.get("validation", {}).get("reason"))
    return 0 if ok == len(results) and ok > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
