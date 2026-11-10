#!/usr/bin/env python3
"""ツール併用ランナー。モデルは「ケース条件の抽出」だけを行い、金額計算は
決定論ソルバ（bench/solve.py）が確定する。算術ミスを構造的に排除する構成。

モデルに抽出スキーマを渡し、JSONで条件を返させる。抽出できた条件から
solve() で値を確定し、tasks.py の正解と照合する（grade.py の amount 相当）。

使い方:
  python3 bench/run_tool.py --models google/gemini-3.5-flash --all --rag --tag tool-rag
  python3 bench/run_local.py --tool --adapter data/distill/adapter --rag --tag tool-local
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.run import (  # noqa: E402
    TASKS, load_articles, Retriever, build_context, chat, load_pricing,
)
from bench.solve import SLOTS, SLOT_DESC, solve  # noqa: E402

EXTRACT_SYS = (
    "あなたは日本の法令実務の入力整理担当である。問題文から、計算に必要な"
    "ケース条件だけを抽出し、JSONで返す。計算はしない。数値は単位を外した数で書く。"
)


def extract_prompt(kind: str, question: str, context: str) -> str:
    keys = SLOTS[kind]
    spec = "\n".join(f'  "{k}": {SLOT_DESC[k]}' for k in keys)
    ctx = f"# 参考条文\n{context}\n\n" if context else ""
    return (
        f"{ctx}# 質問\n{question}\n\n"
        f"# 抽出\n次のJSONだけを出力せよ（キーは固定、値は数値）:\n{{\n{spec}\n}}"
    )


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 単一引用符・末尾カンマの軽い補正
    s = raw.replace("'", '"')
    s = re.sub(r",\s*}", "}", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # キー間のカンマ欠落（"a": 1\n "b": 2）を補う。重ならないので繰り返す。
    s2 = s
    for _ in range(10):
        nxt = re.sub(r'("?\w+"?\s*:\s*[^,{\n]+?)\s*\n\s*("?\w+"?\s*:)', r"\1,\n\2", s2)
        if nxt == s2:
            break
        s2 = nxt
    s2 = re.sub(r",\s*}", "}", s2)
    # クォートなしキー（{a: 1}）を補う
    s2 = re.sub(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)', r'\1"\2"\3', s2)
    try:
        return json.loads(s2)
    except Exception:
        return None


def run_one(task: dict, model: str, retriever, pricing, articles) -> dict:
    kind = task["kind"]
    ctx = ""
    if retriever:
        ctx = build_context(task, retriever, articles or [])
    user = extract_prompt(kind, task["prompt"], ctx)
    rec = {"id": task["id"], "kind": kind, "industry": task["industry"], "model": model}
    try:
        msgs = [{"role": "system", "content": EXTRACT_SYS},
                {"role": "user", "content": user}]
        r = chat(model, msgs, max_tokens=600, temperature=0.0)
    except Exception as e:
        rec.update({"correct": False, "error": str(e)[:300]})
        return rec

    slots = parse_json(r["text"])
    if not slots:
        rec.update({"correct": False, "error": "no-json", "pred": r["text"][:300]})
        return rec
    try:
        val = solve(kind, slots)
    except Exception as e:
        rec.update({"correct": False, "error": f"solve:{e}", "slots": slots})
        return rec

    want = float(task["answer"])
    tol = float(task.get("tol", 1.0))
    ok = abs(float(val) - want) <= tol
    u = r.get("usage", {}) or {}
    p = pricing.get(model, {})
    cost = ((u.get("prompt_tokens", 0) * p.get("in", 0)
             + u.get("completion_tokens", 0) * p.get("out", 0)) / 1_000_000) if p else None
    rec.update({"correct": ok, "want": want, "got": val, "slots": slots,
                "latency_s": round(r["latency_s"], 3), "cost_usd": cost})
    return rec


def summarize(rows: list[dict], model: str) -> dict:
    ok = [r for r in rows if r.get("correct")]
    err = [r for r in rows if r.get("error")]
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    by_ind, by_kind = {}, {}
    for r in rows:
        for b, k in ((by_ind, r["industry"]), (by_kind, r["kind"])):
            d = b.setdefault(k, {"n": 0, "ok": 0})
            d["n"] += 1
            d["ok"] += 1 if r.get("correct") else 0
    return {
        "model": model, "n": len(rows),
        "accuracy": round(len(ok) / len(rows), 4) if rows else 0,
        "errors": len(err),
        "latency_median_s": round(statistics.median(lat), 3) if lat else None,
        "cost_total_usd": round(sum(costs), 6) if costs else None,
        "by_industry": {k: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 3)}
                        for k, v in sorted(by_ind.items())},
        "by_kind": {k: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 3)}
                    for k, v in sorted(by_kind.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--industry", default="")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--tag", default="tool")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--tasks", default=TASKS)
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    if a.industry and not a.all:
        tasks = [t for t in tasks if t["industry"] == a.industry]

    pricing = load_pricing()
    all_articles = load_articles()
    industries = sorted({t["industry"] for t in tasks})
    retrievers = {}
    if a.rag:
        for ind in industries:
            retrievers[ind] = Retriever([x for x in all_articles if x.get("industry") == ind])

    outdir = os.path.join(ROOT, "results", a.tag)
    os.makedirs(outdir, exist_ok=True)
    summaries = []
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        t0 = time.time()
        rows = []
        path = os.path.join(outdir, re.sub(r"[^A-Za-z0-9._-]", "_", model) + ".jsonl")
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex, open(path, "w") as f:
            def submit(t):
                if a.rag:
                    ind = t["industry"]
                    arts = [x for x in all_articles if x.get("industry") == ind]
                    return ex.submit(run_one, t, model, retrievers[ind], pricing, arts)
                return ex.submit(run_one, t, model, None, pricing, None)
            futs = {submit(t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    acc = sum(1 for x in rows if x.get("correct")) / len(rows)
                    print(f"[{model}] {i}/{len(tasks)} acc={acc:.3f}", flush=True)
        s = summarize(rows, model)
        s["wall_s"] = round(time.time() - t0, 1)
        s["rag"] = bool(a.rag)
        s["mode"] = "tool"
        summaries.append(s)
        print(json.dumps(s, ensure_ascii=False, indent=1), flush=True)

    spath = os.path.join(outdir, "summary.json")
    merged = {}
    if os.path.exists(spath):
        for x in json.load(open(spath)):
            merged[x["model"]] = x
    for x in summaries:
        merged[x["model"]] = x
    json.dump(list(merged.values()), open(spath, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {spath}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
