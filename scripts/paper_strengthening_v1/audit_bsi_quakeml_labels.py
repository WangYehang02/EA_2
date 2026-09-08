#!/usr/bin/env python
"""Audit BSI QuakeML packs: event/origin/pick/arrival linkage and S-label provenance.

Does NOT run models or select waveforms by model confidence.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts/results/paper_strengthening_v1/bsi_catalog/raw"
OUT = ROOT / "artifacts/results/paper_strengthening_v1/bsi_catalog"
REPORT = ROOT / "reports/paper_strengthening_v1"

NS = {
    "q": "http://quakeml.org/xmlns/quakeml/1.2",
    "bed": "http://quakeml.org/xmlns/bed/1.2",
}


def local(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text(el, path: str) -> str | None:
    if el is None:
        return None
    # try with and without namespace
    for xpath in (path, path.replace("bed:", "{http://quakeml.org/xmlns/bed/1.2}")):
        hit = el.find(xpath)
        if hit is not None and hit.text:
            return hit.text.strip()
    # brute
    want = path.split(":")[-1]
    for c in el.iter():
        if local(c.tag) == want and c.text and c.text.strip():
            # only first matching direct-ish
            pass
    return None


def audit_zip(path: Path, *, max_events: int = 5000) -> dict:
    picks_phase = Counter()
    pick_eval = Counter()
    arrival_phase = Counter()
    authors = Counter()
    n_events = 0
    n_origins = 0
    n_picks = 0
    n_arrivals = 0
    n_s_picks = 0
    n_s_with_arrival = 0
    n_s_manual_like = 0
    n_s_automatic_like = 0
    n_s_unknown_status = 0
    station_keys = Counter()
    samples = []

    with zipfile.ZipFile(path, "r") as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith((".xml", ".qml")) or "quakeml" in n.lower()]
        if not xml_names:
            xml_names = zf.namelist()[:1]
        # BSI packs often one big xml or many; stream members
        for name in xml_names:
            with zf.open(name) as f:
                # iterative parse
                try:
                    context = ET.iterparse(f, events=("end",))
                except Exception as e:
                    return {"path": str(path), "ok": False, "error": f"iterparse:{e}"}
                for _ev, el in context:
                    tag = local(el.tag)
                    if tag == "event":
                        n_events += 1
                        if n_events > max_events:
                            el.clear()
                            break
                    elif tag == "origin":
                        n_origins += 1
                    elif tag == "pick":
                        n_picks += 1
                        phase = None
                        emode = None
                        author = None
                        net = sta = loc = cha = None
                        time = None
                        for c in el:
                            ct = local(c.tag)
                            if ct == "phaseHint" and c.text:
                                phase = c.text.strip()
                            elif ct == "evaluationMode" and c.text:
                                emode = c.text.strip()
                            elif ct == "creationInfo":
                                for cc in c:
                                    if local(cc.tag) == "author" and cc.text:
                                        author = cc.text.strip()
                            elif ct == "waveformID":
                                net = c.attrib.get("networkCode")
                                sta = c.attrib.get("stationCode")
                                loc = c.attrib.get("locationCode")
                                cha = c.attrib.get("channelCode")
                            elif ct == "time":
                                for cc in c:
                                    if local(cc.tag) == "value" and cc.text:
                                        time = cc.text.strip()
                        picks_phase[phase or "MISSING"] += 1
                        pick_eval[emode or "MISSING"] += 1
                        if author:
                            authors[author] += 1
                        if phase and phase.upper().startswith("S"):
                            n_s_picks += 1
                            station_keys[f"{net}.{sta}.{loc}.{cha}"] += 1
                            if emode is None:
                                n_s_unknown_status += 1
                            elif emode.lower() == "manual":
                                n_s_manual_like += 1
                            elif emode.lower() == "automatic":
                                n_s_automatic_like += 1
                            else:
                                n_s_unknown_status += 1
                            if len(samples) < 20:
                                samples.append(
                                    {
                                        "phase": phase,
                                        "evaluationMode": emode,
                                        "author": author,
                                        "waveformID": f"{net}.{sta}.{loc}.{cha}",
                                        "time": time,
                                    }
                                )
                    elif tag == "arrival":
                        n_arrivals += 1
                        phase = None
                        for c in el:
                            if local(c.tag) == "phase" and c.text:
                                phase = c.text.strip()
                        arrival_phase[phase or "MISSING"] += 1
                        if phase and phase.upper().startswith("S"):
                            n_s_with_arrival += 1
                    # free memory
                    if tag in {"pick", "arrival", "origin", "event"}:
                        el.clear()

    return {
        "path": str(path),
        "ok": True,
        "n_events_seen": n_events,
        "n_origins_seen": n_origins,
        "n_picks_seen": n_picks,
        "n_arrivals_seen": n_arrivals,
        "n_s_picks": n_s_picks,
        "n_s_arrivals": n_s_with_arrival,
        "n_s_evaluationMode_manual": n_s_manual_like,
        "n_s_evaluationMode_automatic": n_s_automatic_like,
        "n_s_evaluationMode_missing_or_other": n_s_unknown_status,
        "pick_phase_counts": dict(picks_phase.most_common(20)),
        "pick_evaluationMode_counts": dict(pick_eval.most_common(20)),
        "arrival_phase_counts": dict(arrival_phase.most_common(20)),
        "top_authors": authors.most_common(15),
        "n_unique_s_waveform_ids": len(station_keys),
        "sample_s_picks": samples,
        "filter_rule_draft": {
            "bulletin_note": "BSI analysts revised events with M>=1.5; bulletin S pick counts are NOT equal to usable 3C records.",
            "do_not_treat_all_S_as_manual": True,
            "proposed": [
                "Keep picks with phaseHint starting with S",
                "Prefer evaluationMode==manual when present",
                "If evaluationMode missing: mark status=UNCONFIRMED_REVIEW_STATUS and list separately until author/bulletin rule verified",
                "Require waveformID net/sta present",
                "Require pick time parseable",
            ],
        },
        "capped_at_events": max_events,
    }


def main() -> int:
    rows = []
    for z in sorted(RAW.glob("*QML*.zip")):
        print("auditing", z.name, flush=True)
        # cap per zip for runtime; full count still informative under cap note
        rows.append(audit_zip(z, max_events=3000))
    (OUT / "quakeml_label_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    # markdown summary
    lines = ["# BSI QuakeML label provenance audit\n", "Parsed from downloaded official QML zips. **No model scores.**\n"]
    lines.append("| ZIP | events(seen≤cap) | S picks | S manual | S auto | S status missing/other | unique S wfIDs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {Path(r['path']).name} | {r.get('n_events_seen')} | {r.get('n_s_picks')} | "
            f"{r.get('n_s_evaluationMode_manual')} | {r.get('n_s_evaluationMode_automatic')} | "
            f"{r.get('n_s_evaluationMode_missing_or_other')} | {r.get('n_unique_s_waveform_ids')} |"
        )
    lines.append("\n## Filter rule (draft, auditable)\n")
    lines.append("- Bulletin S counts ≠ final usable three-component records.\n")
    lines.append("- Do **not** auto-accept every phaseHint=S as manual.\n")
    lines.append("- Prefer `evaluationMode=manual`; missing mode → `UNCONFIRMED_REVIEW_STATUS` bucket.\n")
    (REPORT / "BSI_LABEL_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "quakeml_label_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
