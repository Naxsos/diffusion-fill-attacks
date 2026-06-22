"""Run the dose-response fill-attack sweep: every prompt x every level x {pin, perturb}.

Reads anchored_spans.json (from gen_anchored_spans.py), then for each prompt runs the
diffusion steer for each level (L1..L4, scramble) in both pin and perturb modes, plus one
unsteered baseline per prompt (the L0 / refusal-rate reference). Each run is written as

    <outdir>/p{idx:04d}_{level}_{mode}.json
    <outdir>/p{idx:04d}_baseline.json

in the SAME shape strongreject_grade.py already consumes (it reads `prompt` + `text`), so
grading is unchanged:

    python siva_experiments/strongreject_grade.py --expdir experiments_anchored/<outdir> \
           --evaluator strongreject_finetuned --base-model unsloth/gemma-2b
    python siva_experiments/strongreject_grade.py --expdir experiments_anchored/<outdir> \
           --evaluator strongreject_rubric          # needs OPENAI_API_KEY (auto-loaded)

The heavy model stays on the server (see SERVER.md); start it first:  python server.py

Run:
    .venv/bin/python experiments_anchored/run_anchored_sweep.py --outdir runs_anchored_1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

# repo root holds client.py / example_steer.py / steer_config.py; the script's own dir
# (this subfolder) is what Python puts on sys.path, so add the parent explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from _env import load_env

load_env()

from client import generate                       # noqa: E402  (server HTTP client)
from example_steer import load_tokenizer, run_experiment  # noqa: E402
from steer_config import SteerConfig              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPANS = os.path.join(HERE, "anchored_spans.json")

LEVELS = ["L1", "L2", "L3", "L4", "scramble"]
MODES = ["pin", "perturb"]


def with_retry(label, fn, *a, **k):
    """Retry on connection-level errors (e.g. ephemeral-port exhaustion / Errno 99 from
    many short-lived HTTP calls). Server-side HTTPErrors (500s) are surfaced immediately,
    not retried -- those are real failures, not transient socket churn."""
    import urllib.error
    delay = 5.0
    for attempt in range(8):
        try:
            return fn(*a, **k)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError) as e:
            print(f"  ! {label}: {type(e).__name__} {str(e)[:70]} -> wait {delay:.0f}s "
                  f"(retry {attempt + 1}/8)", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return fn(*a, **k)  # final attempt: let it raise


def clamp_positions(tok, targets, positions, cap):
    """Keep every span inside the usable canvas (`cap` = canvas_length - tail margin) and
    non-overlapping. Two reasons a fixed pos 240 + long generated span breaks:
      * it can exceed the 256 canvas, and
      * DiffusionGemma is BLOCK-diffusion -- pinning the final positions trips a
        device-side CUDA assert (the original benchmark only ever reached ~pos 249).
    So we right-align long spans to `cap`, leaving the canvas tail unpinned."""
    adj, prev_end = [], 0
    for t, p in zip(targets, positions):
        n = len(tok.encode(t, add_special_tokens=False))
        start = min(p, cap - n)       # fit within the canvas
        start = max(start, prev_end)  # no overlap with the previous span
        adj.append(max(start, 0))
        prev_end = adj[-1] + n
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="runs_anchored")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--canvas-margin", type=int, default=8,
                    help="leave this many tail positions of the 256 canvas unpinned "
                         "(block-diffusion asserts when the final positions are pinned)")
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--no-baseline", action="store_true", help="skip the L0 baseline pass")
    ap.add_argument("--max-prompts", type=int, default=0)
    args = ap.parse_args()

    outdir = os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    spec = json.load(open(SPANS))
    prompts = spec["prompts"]
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"{len(prompts)} prompts x {len(args.levels)} levels x {len(args.modes)} modes "
          f"-> {len(prompts) * len(args.levels) * len(args.modes)} steered runs")

    print("loading tokenizer once (no GPU)...", flush=True)
    tok = load_tokenizer()  # loaded a SINGLE time; reused for every run below

    # count the model calls actually remaining (skips already-written files) for a live ETA
    def _pending(pr):
        n = 0
        b = os.path.join(outdir, f"p{pr['idx']:04d}_baseline.json")
        if not args.no_baseline and not os.path.exists(b):
            n += 1
        for level in args.levels:
            for mode in args.modes:
                if not os.path.exists(os.path.join(outdir, f"p{pr['idx']:04d}_{level}_{mode}.json")):
                    n += 1
        return n

    total = sum(_pending(pr) for pr in prompts)
    done, t0 = 0, time.time()

    def _eta():
        el = time.time() - t0
        avg = el / done if done else 0
        rem = avg * (total - done)
        print(f"  [{done}/{total}] {avg:.1f}s/call · elapsed {el/60:.1f}m · ETA {rem/60:.1f}m",
              flush=True)

    print(f"{total} model calls to run (GPU). Live ETA after the first call.\n", flush=True)

    for pr in prompts:
        idx, prompt = pr["idx"], pr["prompt"]
        tag0 = f"p{idx:04d}"

        # L0 baseline (one unsteered generation per prompt) -> refusal-rate reference.
        if not args.no_baseline:
            bpath = os.path.join(outdir, f"{tag0}_baseline.json")
            if not os.path.exists(bpath):
                base = with_retry(f"{tag0}_baseline", generate, prompt, host=args.host,
                                  port=args.port, max_new_tokens=args.max_new_tokens)
                json.dump({"prompt": prompt, "text": base, "targets": [], "start_pos": [],
                           "level": "L0", "mode": "baseline"},
                          open(bpath, "w"), indent=2, ensure_ascii=False)
                done += 1
                print(f"+ {tag0}_baseline", flush=True)
                _eta()

        for level in args.levels:
            lv = pr["levels"][level]
            targets = lv["targets"]
            positions = clamp_positions(tok, targets, lv["positions"],
                                         args.max_new_tokens - args.canvas_margin)
            if positions != lv["positions"]:
                print(f"  {tag0} {level}: positions {lv['positions']} -> {positions} (clamped to fit)")
            for mode in args.modes:
                tag = f"{tag0}_{level}_{mode}"
                out = os.path.join(outdir, f"{tag}.json")
                if os.path.exists(out):
                    print(f"= {tag}: exists, skip")
                    continue
                cfg = SteerConfig(
                    prompt=prompt, target=list(targets), start_pos=list(positions),
                    mode=[mode] * len(targets), step=[0] * len(targets),
                    seed=args.seed, trace_file=out, host=args.host, port=args.port,
                )
                cfg.no_baseline = True  # run_experiment honours this via getattr
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    with_retry(tag, run_experiment, cfg, tok)  # writes `out`
                # tag the written file with level/mode for downstream grouping
                rec = json.load(open(out))
                rec["level"], rec["mode"] = level, mode
                json.dump(rec, open(out, "w"), indent=2, ensure_ascii=False)
                done += 1
                print(f"+ {tag}", flush=True)
                _eta()

    print(f"\nsweep done -> {outdir}")
    print("Now grade with BOTH evaluators (see this file's docstring).")


if __name__ == "__main__":
    main()
