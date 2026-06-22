"""Resilient StrongREJECT *rubric* grading: one bad judge reply -> NaN, not a dead run.

strong_reject's `evaluate_dataset(..., ["strongreject_rubric"])` runs the gpt-4o-mini judge
over the whole dataset via `datasets.map` with multiprocessing; if ANY row's judge reply
fails to parse (e.g. missing 'refusal' field), the entire batch raises `KeyError`. At n=660
that happens often enough to lose the run. This grader chunks the work and, on a chunk
failure, falls back to row-by-row so only the genuinely-unparseable rows become NaN.

Writes the SAME CSV schema as siva_experiments/strongreject_grade.py so the analysis /
report scripts read it unchanged.

    .venv/bin/python experiments_anchored/grade_rubric_resilient.py \
        --expdir experiments_anchored/runs_anchored \
        --out    experiments_anchored/runs_anchored/scores_rubric.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re

from _env import load_env

load_env()  # OPENAI_API_KEY from ../.env

FNAME_RE = re.compile(r"^p(\d+)_(.+)\.json$")
EXTRA = ("refusal", "convincingness", "specificity")


def score_rows(rows, chunk, eval_kwargs=None):
    """Return list of dicts {score, refusal?, ...} aligned to rows; NaN on unparseable."""
    import json as _json

    from datasets import Dataset
    from strong_reject.evaluate import evaluate_dataset

    eval_kwargs = eval_kwargs or {}

    def run(batch):
        ds = Dataset.from_dict({"forbidden_prompt": [r[2] for r in batch],
                                "response": [r[3] for r in batch]})
        scored = evaluate_dataset(ds, ["strongreject_rubric"], **eval_kwargs)
        cols = {c: scored[c] for c in ("score", *EXTRA) if c in scored.column_names}
        return [{k: cols[k][i] for k in cols} for i in range(len(batch))]

    out = []
    for i in range(0, len(rows), chunk):
        batch = rows[i:i + chunk]
        try:
            out.extend(run(batch))
            print(f"  chunk {i}-{i + len(batch)}: ok", flush=True)
        except Exception as e:  # noqa: BLE001 - fall back to row-by-row
            print(f"  chunk {i}-{i + len(batch)}: {type(e).__name__} -> row-by-row", flush=True)
            for r in batch:
                try:
                    out.extend(run([r]))
                except Exception as e2:  # noqa: BLE001 - this single row is unparseable
                    print(f"    row p{r[0]:04d}_{r[1]}: FAILED ({type(e2).__name__}) -> NaN", flush=True)
                    out.append({"score": float("nan")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--judge-model", nargs="+", default=None,
                    help="litellm judge model id(s), tried in order (default: gpt-4o-mini).")
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.expdir, "scores_rubric.csv")

    eval_kwargs = {}
    if args.judge_model:
        eval_kwargs["models"] = args.judge_model
    if args.temperature is not None:
        eval_kwargs["temperature"] = args.temperature

    rows = []  # (pidx, strat, prompt, response)
    for fn in sorted(glob.glob(os.path.join(args.expdir, "p*_*.json"))):
        m = FNAME_RE.match(os.path.basename(fn))
        if not m:
            continue
        import json
        d = json.load(open(fn))
        rows.append((int(m.group(1)), m.group(2), d["prompt"], d.get("text", "")))
    print(f"loaded {len(rows)} responses from {args.expdir}", flush=True)

    scored = score_rows(rows, args.chunk, eval_kwargs)
    present_extra = [c for c in EXTRA if any(c in s for s in scored)]

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_idx", "strategy", "score", *present_extra, "forbidden_prompt", "response"])
        for (pidx, strat, prompt, resp), s in zip(rows, scored):
            sc = s.get("score", float("nan"))
            w.writerow([pidx, strat, f"{sc:.4f}", *[s.get(c, "") for c in present_extra], prompt, resp])

    n_ok = sum(1 for s in scored if not math.isnan(s.get("score", float("nan"))))
    print(f"wrote {out}  ({n_ok}/{len(rows)} graded, {len(rows) - n_ok} NaN)")


if __name__ == "__main__":
    main()
