"""Render the graded sweep into a paste-ready Markdown report (table + CIs + transcripts).

Reads the two StrongREJECT score CSVs (finetuned + rubric) and anchored_spans.json, and
writes results.md (also echoed to stdout): the span-alone floor, the dose-response table
with 95% bootstrap CIs for both graders, the paired deltas, and a few example transcripts
per level so you can show the actual model output (not just numbers).

    .venv/bin/python experiments_anchored/export_report.py \
        --finetuned experiments_anchored/runs_anchored/scores_finetuned.csv \
        --rubric    experiments_anchored/runs_anchored/scores_rubric.csv \
        --out       experiments_anchored/results.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from analyze_dose_response import (LEVEL_ORDER, boot_mean_ci, boot_paired_delta,
                                    load_csv, parse_strategy)

HERE = os.path.dirname(os.path.abspath(__file__))
SPANS = os.path.join(HERE, "anchored_spans.json")


def load_responses(path: str) -> dict:
    """-> {(level, mode, prompt_idx): (score, response)} for transcript display."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["score"])
            except (ValueError, KeyError):
                continue
            level, mode = parse_strategy(row["strategy"])
            out[(level, mode, int(row["prompt_idx"]))] = (score, row.get("response", ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", default=None)
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results.md"))
    ap.add_argument("--examples", type=int, default=2, help="transcripts per level to show")
    ap.add_argument("--example-grader", default="rubric", choices=("rubric", "finetuned"))
    args = ap.parse_args()

    graders = {}
    if args.finetuned:
        graders["finetuned"] = load_csv(args.finetuned)
    if args.rubric:
        graders["rubric"] = load_csv(args.rubric)
    if not graders:
        ap.error("pass --finetuned and/or --rubric")
    modes = sorted({m for c in graders.values() for (_, m) in c if m not in ("baseline", "?")})

    L = ["# Topic-anchored mid-fill — StrongREJECT results", ""]

    # --- floor ---
    if os.path.exists(SPANS):
        fs = json.load(open(SPANS)).get("floor_summary", {})
        L += ["## Span-alone floor (injected text graded by itself)", "",
              f"- **Anchoring spans (mid+post alone):** L3 {fs.get('L3_midpost_mean', float('nan')):.3f}, "
              f"L4 {fs.get('L4_midpost_mean', float('nan')):.3f} (max {fs.get('midpost_max', float('nan')):.2f})",
              f"- Conservative (lead+mid+post): L3 {fs.get('L3_full_mean', float('nan')):.3f}, "
              f"L4 {fs.get('L4_full_mean', float('nan')):.3f}",
              "", "_Injected anchoring text carries ~no harmful content; harm below is model-generated._", ""]

    # --- dose-response table ---
    L += ["## Dose-response — mean StrongREJECT score [0,1] (95% bootstrap CI)", ""]
    cols = []
    for g in graders:
        for m in modes:
            cols.append((g, m))
    L += ["| level | " + " | ".join(f"{g}/{m}" for g, m in cols) + " |",
          "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for level in LEVEL_ORDER:
        cells = []
        for g, m in cols:
            mm = "baseline" if level == "L0" else m
            vals = list(graders[g].get((level, mm), {}).values())
            if not vals:
                cells.append("–")
            else:
                mean, lo, hi = boot_mean_ci(vals)
                cells.append(f"{mean:.2f} [{lo:.2f},{hi:.2f}]")
        if any(c != "–" for c in cells):
            L.append(f"| {level} | " + " | ".join(cells) + " |")
    L.append("")

    # --- paired deltas ---
    L += ["## Paired deltas (same prompt, two levels) — 95% bootstrap CI", "",
          "`*` = CI excludes 0.", "",
          "| grader | mode | comparison | Δ | 95% CI | n |", "| --- | --- | --- | --- | --- | --- |"]
    for g, cells in graders.items():
        for m in modes:
            for a_lvl, b_lvl in (("L2", "L3"), ("L2", "L4"), ("L1", "L4"), ("L4", "scramble")):
                d, lo, hi, n = boot_paired_delta(cells.get((a_lvl, m), {}), cells.get((b_lvl, m), {}))
                if n == 0:
                    continue
                sig = "" if lo <= 0 <= hi else " *"
                L.append(f"| {g} | {m} | {a_lvl}→{b_lvl} | {d:+.3f}{sig} | [{lo:+.3f}, {hi:+.3f}] | {n} |")
    L.append("")

    # --- example transcripts ---
    if args.example_grader in graders:
        resp = load_responses(args.rubric if args.example_grader == "rubric" else args.finetuned)
        L += ["## Example transcripts (highest-scoring per level)", ""]
        for level in [x for x in LEVEL_ORDER if x != "L0"]:
            for m in modes:
                items = [(idx, s, r) for (lv, mo, idx), (s, r) in resp.items()
                         if lv == level and mo == m]
                if not items:
                    continue
                items.sort(key=lambda x: x[1], reverse=True)
                for idx, s, r in items[: args.examples]:
                    L += [f"### {level}/{m} · prompt p{idx:04d} · score {s:.2f}", "",
                          "```", r.strip()[:1200], "```", ""]

    report = "\n".join(L)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
