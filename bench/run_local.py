#!/usr/bin/env python3
"""仕立て後モデル（LoRAアダプタ）をローカルMLXで推論し、同じ採点器で実測する。

base と同じ RAG 形式（# 参考条文 + # 質問）で問い、grade.py で決定論的に採点する。
結果は results/shitate/<name>.jsonl と summary.json に保存する。

使い方:
  python3 bench/run_local.py --model mlx-community/Qwen3-8B-4bit \
      --adapter data/distill/adapter --rag --tag shitate-rag
  python3 bench/run_local.py --model mlx-community/Qwen3-8B-4bit \
      --adapter data/distill/adapter --tag shitate-base
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.run import (  # noqa: E402
    TASKS, load_articles, Retriever, build_context, SYSTEM, SYSTEM_RAG,
)
from bench.grade import grade  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--industry", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="shitate")
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--max-tokens", type=int, default=1500)
    a = ap.parse_args()

    from mlx_lm import load, generate

    print(f"loading {a.model} adapter={a.adapter or '(none)'}", file=sys.stderr)
    model, tok = load(a.model, adapter_path=a.adapter or None)

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    if a.industry and not a.all:
        tasks = [t for t in tasks if t["industry"] == a.industry]
    if a.limit:
        import random
        tasks = random.Random(0).sample(tasks, min(a.limit, len(tasks)))

    all_articles = load_articles()
    industries = sorted({t["industry"] for t in tasks})
    retrievers = {}
    if a.rag:
        for ind in industries:
            retrievers[ind] = Retriever([x for x in all_articles if x.get("industry") == ind])

    outdir = os.path.join(ROOT, "results", a.tag)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "local.jsonl")
    rows = []
    t0 = time.time()
    with open(path, "w") as f:
        for i, t in enumerate(tasks, 1):
            system = SYSTEM_RAG if a.rag else SYSTEM
            user = t["prompt"]
            if a.rag:
                arts = [x for x in all_articles if x.get("industry") == t["industry"]]
                ctx = build_context(t, retrievers[t["industry"]], arts)
                user = f"# 参考条文\n{ctx}\n\n# 質問\n{t['prompt']}"
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             enable_thinking=False)
            st = time.time()
            text = generate(model, tok, prompt=prompt, max_tokens=a.max_tokens,
                            verbose=False)
            dt = time.time() - st
            ok, why = grade(t, text)
            rec = {"id": t["id"], "kind": t["kind"], "industry": t["industry"],
                   "correct": ok, "why": why, "answer": t["answer"],
                   "pred": text[:600], "latency_s": round(dt, 3)}
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                acc = sum(1 for r in rows if r["correct"]) / len(rows)
                print(f"[{a.tag}] {i}/{len(tasks)} acc={acc:.3f}", file=sys.stderr)

    ok = [r for r in rows if r["correct"]]
    by_ind: dict[str, list[int]] = {}
    by_kind: dict[str, list[int]] = {}
    for r in rows:
        by_ind.setdefault(r["industry"], []).append(1 if r["correct"] else 0)
        by_kind.setdefault(r["kind"], []).append(1 if r["correct"] else 0)
    summary = {
        "model": a.model, "adapter": a.adapter, "rag": a.rag, "tag": a.tag,
        "n": len(rows), "accuracy": round(len(ok) / max(1, len(rows)), 4),
        "wall_s": round(time.time() - t0, 1),
        "by_industry": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
                        for k, v in by_ind.items()},
        "by_kind": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
                    for k, v in by_kind.items()},
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
