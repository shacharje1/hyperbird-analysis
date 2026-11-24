"""Experimental design, read from design.json (in the working folder, or the path in $HB_DESIGN).

Every script takes the folder -> references / time point / scan session map, the sample-name pattern, the
treatment groups, the planned contrasts and the infection (inoculated vs mock) pair from here, so nothing about
one experiment is hard-coded. See design_template.json for the fields, design_vg3.json for a filled example.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#d14b8f", "#4bb5c9", "#7a7a7a"]


def _path():
    p = Path(os.environ.get("HB_DESIGN", "design.json"))
    if not p.exists():
        sys.exit(f"No experimental design found ({p.resolve()}). Create design.json first (copy design_template.json "
                 "and fill it in with the user: folders, time points, sessions, groups, contrasts, mock pair), "
                 "or point $HB_DESIGN at it.")
    return p


D = json.loads(_path().read_text())

FOLDERS = {f["path"]: f for f in D["folders"]}
REFS = {p: (f["white"], f["black"]) for p, f in FOLDERS.items()}
ANALYSE = {p for p, f in FOLDERS.items() if f.get("analyse", True)}
TIMES = D["times"]
DPIS = TIMES  # older name used throughout the scripts
GROUPS = [g["series"] for g in D["groups"]]
LABELS = {g["series"]: g.get("label", g["series"]) for g in D["groups"]}
COLORS = {g["series"]: g.get("color", PALETTE[i % len(PALETTE)]) for i, g in enumerate(D["groups"])}
TIME_COLORS = dict(zip(TIMES, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#eda100", "#e8417f", "#3fb8c4", "#7a7a7a"]))
SESSIONS = list(dict.fromkeys(f.get("session", f["time"]) for p, f in FOLDERS.items() if p in ANALYSE))
SESSION_COLORS = dict(zip(SESSIONS, PALETTE))
CONTRASTS = [(c["title"], c["a"], c["b"]) for c in D.get("contrasts", [])]
INF = D.get("infection") or {}
POS, NEG = INF.get("inoculated"), INF.get("mock")
INOCULATED = [g["series"] for g in D["groups"] if g.get("inoculated")]
CHANGE = f"Δ {TIMES[0]}→{TIMES[-1]}" if len(TIMES) > 1 else None
NAME_RE = re.compile(D.get("name_regex", r"^(?P<id>\d+)-(?P<group>[^ _]+)[ _](?P<cond>[A-Za-z]+)"))
ALIASES = D.get("aliases", {})
SERIES_FMT = D.get("series_format", "{group} {cond}")


def short(series):
    """Display name: undo aliases that only exist to merge spellings (e.g. nln -> nIn)."""
    for k, v in D.get("display", {}).items():
        series = series.replace(k, v)
    return series


def annotate(d):
    """Add group, cond, series, id, dpi (= time point), session from folder + sample name; keep analysed folders."""
    d = d[d.folder.isin(ANALYSE)].copy()
    lab = d.name.str.extract(NAME_RE)
    for c in ("group", "cond"):
        old = d[c] if c in d else None
        d[c] = lab[c] if old is None else old.where(old.notna(), lab[c])
        d[c] = d[c].astype(str).str.replace("_", " ").str.strip().replace(ALIASES)
    d["series"] = [SERIES_FMT.format(group=g, cond=c) for g, c in zip(d.group, d.cond)]
    d["id"] = lab["id"].astype(int).values
    d["dpi"] = d.folder.map(lambda f: FOLDERS[f]["time"])
    d["session"] = d.folder.map(lambda f: FOLDERS[f].get("session", FOLDERS[f]["time"]))
    unknown = sorted(set(d.series) - set(GROUPS))
    if unknown:
        print(f"[design] samples whose series is not in design groups (excluded): {unknown}", file=sys.stderr)
        d = d[d.series.isin(GROUPS)]
    return d


def paired_change(wide, ix):
    """Per-disk change from the first to the last time point (wide = pivot with (index, time) columns)."""
    return wide[(ix, TIMES[-1])] - wide[(ix, TIMES[0])]


def need(cond, what):
    if not cond:
        sys.exit(f"design.json has no {what}; this analysis needs it.")


def describe():
    print(f"Experiment: {D.get('experiment', '?')}")
    print(f"  times {TIMES}, sessions {SESSIONS}")
    for g in D["groups"]:
        print(f"  {g['series']:<10} {g.get('label', '')}  inoculated={g.get('inoculated')}")
    print(f"  contrasts: {[c[0] for c in CONTRASTS]}")
    print(f"  infection pair: {POS} vs {NEG}")


if __name__ == "__main__":
    describe()
