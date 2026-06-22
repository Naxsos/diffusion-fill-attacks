"""Dose-response analysis with PAIRED bootstrap CIs -- the figure the blog post needs.

Reads the StrongREJECT score CSVs (one per grader; columns: prompt_idx, strategy, score,
...) that strongreject_grade.py writes, plus anchored_spans.json for the span-alone floor.

Strategy names are `p####_<level>_<mode>` -> grouped into (level, mode); `baseline` -> L0.
For each grader it prints:

  * a dose-response table  level x mode -> mean StrongREJECT score [0,1] with a 95%
    bootstrap CI (resampling prompts), for every grader side by side;
  * PAIRED deltas (same prompt under two levels): L2->L3, L2->L4, L4->scramble, with a
    95% bootstrap CI on the paired difference -- this is what makes n=60 enough, because
    each prompt is its own control;
  * the span-alone floor (mean harmfulness of the INJECTED text by itself) so the post
    can show injected payload stayed near zero while output harmfulness rose.

Grade each run dir with BOTH evaluators into separate CSVs first:
    python siva_experiments/strongreject_grade.py --expdir experiments_anchored/runs_anchored \
        --evaluator strongreject_finetuned --base-model unsloth/gemma-2b \
        --out experiments_anchored/runs_anchored/scores_finetuned.csv
    python siva_experiments/strongreject_grade.py --expdir experiments_anchored/runs_anchored \
        --evaluator strongreject_rubric \
        --out experiments_anchored/runs_anchored/scores_rubric.csv

Then:
    .venv/bin/python experiments_anchored/analyze_dose_response.py \
        --finetuned experiments_anchored/runs_anchored/scores_finetuned.csv \
        --rubric    experiments_anchored/runs_anchored/scores_rubric.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SPANS = os.path.join(HERE, "anchored_spans.json")
LEVEL_ORDER = ["L0", "L1", "L2", "L3", "L4", "scramble"]
N_BOOT = 10000
random.seed(0)


def parse_strategy(strat: str) -> tuple[str, str]:
    if strat == "baseline":
        return "L0", "baseline"
    parts = strat.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("pin", "perturb"):
        return parts[0], parts[1]
    return strat, "?"


def load_csv(path: str) -> dict:
    """-> {(level, mode): {prompt_idx: score}}  (skips non-numeric / nan cells)."""
    cells: dict = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                score = float(row["score"])
            except (ValueError, KeyError):
                continue
            if score != score:  # nan
                continue
            level, mode = parse_strategy(row["strategy"])
            cells[(level, mode)][int(row["prompt_idx"])] = score
    return cells


def boot_mean_ci(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    means = []
    for _ in range(N_BOOT):
        means.append(sum(vals[random.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return sum(vals) / n, means[int(0.025 * N_BOOT)], means[int(0.975 * N_BOOT)]


def boot_paired_delta(a: dict, b: dict) -> tuple[float, float, float, int]:
    """Paired mean(b)-mean(a) over prompts present in BOTH, with 95% bootstrap CI."""
    keys = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in keys]
    if not diffs:
        return float("nan"), float("nan"), float("nan"), 0
    n = len(diffs)
    boots = []
    for _ in range(N_BOOT):
        boots.append(sum(diffs[random.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return sum(diffs) / n, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)], n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetuned", default=None)
    ap.add_argument("--rubric", default=None)
    args = ap.parse_args()

    graders = {}
    if args.finetuned:
        graders["finetuned"] = load_csv(args.finetuned)
    if args.rubric:
        graders["rubric"] = load_csv(args.rubric)
    if not graders:
        ap.error("pass --finetuned and/or --rubric")

    modes = sorted({m for cells in graders.values() for (_, m) in cells if m not in ("baseline", "?")})

    # --- span-alone floor ---
    if os.path.exists(SPANS):
        spec = json.load(open(SPANS))
        fs = spec.get("floor_summary", {})
        nan = float("nan")
        print("=== SPAN-ALONE FLOOR (injected text graded by itself, rubric) ===")
        print(f"  anchoring spans (mid+post alone)  -> L3 {fs.get('L3_midpost_mean', nan):.3f}   "
              f"L4 {fs.get('L4_midpost_mean', nan):.3f}   max {fs.get('midpost_max', nan):.2f}")
        print(f"  conservative (lead+mid+post)      -> L3 {fs.get('L3_full_mean', nan):.3f}   "
              f"L4 {fs.get('L4_full_mean', nan):.3f}   (floor cutoff {spec.get('floor')})")
        print("  Headline defense: anchoring-span floor << full-output mean below.\n")

    # --- dose-response table ---
    for gname, cells in graders.items():
        print(f"=== DOSE-RESPONSE  [grader: {gname}]  StrongREJECT score [0,1], mean (95% CI) ===")
        header = "  level     " + "".join(f"{m:>22}" for m in (["baseline"] if "L0" in {l for l, _ in cells} else []) + modes)
        print(header)
        for level in LEVEL_ORDER:
            row_modes = ["baseline"] if level == "L0" else modes
            cellstr = ""
            for m in row_modes:
                vals = list(cells.get((level, m), {}).values())
                if not vals:
                    cellstr += f"{'-':>22}"
                    continue
                mean, lo, hi = boot_mean_ci(vals)
                cellstr += f"{f'{mean:.2f} [{lo:.2f},{hi:.2f}]':>22}"
            if cellstr.strip():
                print(f"  {level:<8}{cellstr}")
        print()

        # --- paired deltas (the key inferential result) ---
        print(f"  PAIRED deltas [grader: {gname}] (positive => more harmful), 95% bootstrap CI:")
        comparisons = [("L2", "L3"), ("L2", "L4"), ("L1", "L4"), ("L4", "scramble")]
        for m in modes:
            for a_lvl, b_lvl in comparisons:
                a = cells.get((a_lvl, m), {})
                b = cells.get((b_lvl, m), {})
                d, lo, hi, n = boot_paired_delta(a, b)
                if n == 0:
                    continue
                sig = "" if (lo <= 0 <= hi) else "  *"   # CI excludes 0
                print(f"    [{m:<7}] {a_lvl}->{b_lvl}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  n={n}{sig}")
        print()

    print("* = 95% CI excludes 0 (paired difference distinguishable from no effect).")


if __name__ == "__main__":
    main()
