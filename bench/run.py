#!/usr/bin/env python3
"""shitate-bench ランナー。teai (OpenAI互換) 経由で複数モデルを業種別に実測する。

kenchiku-bench のランナーを多業種へ一般化。業種ごとに対象法令の条文だけを
RAG コーパスにする（税務モデルに労基法を混ぜない）。

使い方:
  python3 bench/run.py --models gpt-4.1-nano,claude-haiku-5-5 --industry tax
  python3 bench/run.py --models google/gemini-3.5-flash --industry labor --rag
  python3 bench/run.py --models gpt-4.1-nano --all --rag --tag all-rag

出力: results/<tag>/<model>.jsonl, results/<tag>/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench.grade import grade, kanji_num  # noqa: E402

TASKS = os.path.join(ROOT, "data", "tasks.jsonl")
ARTICLES = os.path.join(ROOT, "corpus", "articles.jsonl")
PRICING_CACHE = os.path.join(ROOT, "data", "pricing.json")

BASE = os.environ.get("TEAI_BASE_URL", "https://api.teai.io/v1")
KEY = os.environ.get("TEAI_API_KEY", "")

SYSTEM = (
    "あなたは日本の法令・税務・労務・金融実務に精通した専門家である。"
    "与えられた問題を、法令の条文と法定の率・計算式に基づいて正確に解く。"
    "推測で数値を書かず、計算は慎重に行い、最終的な答えを明示する。"
    "金額は円単位、率は%で答える。"
)

SYSTEM_RAG = (
    SYSTEM
    + "\n\n参考条文が与えられた場合は、必ずその条文の記述を根拠に答えよ。"
      "参考条文に答えが無い場合は、自分の知識で補ってよいが、その旨を明示せよ。"
)


# ---------------------------------------------------------------- 価格表

def load_pricing() -> dict:
    if os.path.exists(PRICING_CACHE):
        return json.load(open(PRICING_CACHE))
    req = urllib.request.Request(f"{BASE}/models/pricing", headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)["data"]
    out = {m["id"]: {"in": m["input_per_1m_tokens"], "out": m["output_per_1m_tokens"],
                     "ctx": m.get("context_window")} for m in d}
    os.makedirs(os.path.dirname(PRICING_CACHE), exist_ok=True)
    json.dump(out, open(PRICING_CACHE, "w"), ensure_ascii=False, indent=0)
    return out


# ---------------------------------------------------------------- 検索（RAG）

def load_articles(industry: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
    if industry:
        rows = [r for r in rows if r.get("industry") == industry]
    return rows


def _terms(s: str) -> set[str]:
    out = set()
    for m in re.finditer(r"[一-龥]{2,}", s):
        t = m.group(0)
        for i in range(len(t) - 1):
            out.add(t[i:i + 2])
    for m in re.finditer(r"[ァ-ヴー]{2,}", s):
        out.add(m.group(0))
    for m in re.finditer(r"[A-Za-z]{2,}", s):
        out.add(m.group(0).lower())
    return out


def _ngrams(s: str, n: int) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


class Retriever:
    """BM25 + 見出し重み + 文字n-gram一致の簡易ハイブリッド検索（外部依存なし）。"""

    NGRAM = 5

    def __init__(self, articles: list[dict]):
        self.docs = articles
        self.doc_terms = [_terms((a["caption"] + " ") * 3 + a["text"]) for a in articles]
        self.doc_ngrams = [_ngrams(a["caption"] + a["text"], self.NGRAM) for a in articles]
        self.df: dict[str, int] = {}
        for ts in self.doc_terms:
            for t in ts:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(articles)
        self.avgdl = sum(len(t) for t in self.doc_terms) / max(1, self.N)
        self.captions = [re.sub(r"[（）()\s]", "", a["caption"]) for a in articles]

    def search(self, query: str, k: int = 3) -> list[dict]:
        import math
        q = _terms(query)
        qg = _ngrams(query, self.NGRAM)
        qn = re.sub(r"[（）()\s]", "", query)
        law_pref = ""
        for cand in ("所得税法", "法人税法", "消費税法", "労働基準法", "労働契約法",
                     "利息制限法", "出資法"):
            if cand in query:
                law_pref = cand
                break
        scored = []
        for i, ts in enumerate(self.doc_terms):
            if not ts:
                continue
            s = 0.0
            dl = len(ts)
            for t in q:
                if t not in ts:
                    continue
                idf = math.log(1 + (self.N - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                s += idf * 2.2 / (1 + 1.2 * (0.25 + 0.75 * dl / self.avgdl))
            if qg:
                dg = self.doc_ngrams[i]
                inter = len(qg & dg)
                if inter:
                    s += 3.0 * inter / len(qg)
            cap = self.captions[i]
            if cap and len(cap) >= 2 and cap in qn:
                s += 12.0 + 2.0 * len(cap)
            if law_pref and self.docs[i]["law"] == law_pref:
                s *= 1.5
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.docs[i] for _, i in scored[:k]]


ARTICLE_REF = re.compile(r"第\s*([0-9〇一二三四五六七八九十百千]+(?:の[0-9一二三四五六七八九十]+)?)\s*条")


def resolve_refs(query: str, articles: list[dict]) -> list[dict]:
    refs = set()
    for m in ARTICLE_REF.finditer(query):
        a = m.group(1).replace("の二", "の2").replace("の三", "の3").replace("の四", "の4")
        mm = re.fullmatch(r"([〇一二三四五六七八九十百千]+)(の\d+)?", a)
        if mm:
            n = kanji_num(mm.group(1))
            if n is not None:
                a = f"{n}{mm.group(2) or ''}"
        refs.add(a)
    if not refs:
        return []
    law = ""
    for cand in ("所得税法", "法人税法", "消費税法", "労働基準法", "労働契約法",
                 "利息制限法", "出資法"):
        if cand in query:
            law = cand
            break
    out = []
    for a in sorted(refs):
        for r in articles:
            if (not law or r["law"] == law) and r["article"] == a:
                out.append(r)
                break
    return out


def rag_context(hits: list[dict], max_chars: int = 6000) -> str:
    parts, used = [], 0
    for a in hits:
        chunk = f"【{a['law']}第{a['article']}条 {a['caption']}】" + a["text"]
        if used + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - used)]
        if not chunk:
            break
        parts.append(chunk)
        used += len(chunk)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


# ---------------------------------------------------------------- API

def chat(model: str, messages: list[dict], max_tokens: int = 2500,
         temperature: float = 0.0, retries: int = 4) -> dict:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    if os.environ.get("NO_THINK"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{BASE}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.load(r)
            dt = time.time() - t0
            msg = j["choices"][0]["message"]
            return {"msg": msg, "text": msg.get("content") or msg.get("reasoning") or "",
                    "usage": j.get("usage", {}), "latency_s": dt,
                    "finish": j["choices"][0].get("finish_reason"), "retries": attempt}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last) from None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last) from None
    raise RuntimeError(last or "unknown")


# ---------------------------------------------------------------- 実行

def build_context(task: dict, retriever: Retriever, articles: list[dict]) -> str:
    direct = resolve_refs(task["prompt"], articles)
    hits = retriever.search(task["prompt"], k=4)
    seen = {(a["law"], a["article"]) for a in direct}
    merged = direct + [h for h in hits if (h["law"], h["article"]) not in seen]
    return rag_context(merged)


def run_one(task: dict, model: str, retriever: Retriever | None, pricing: dict,
            articles: list[dict] | None = None) -> dict:
    system = SYSTEM_RAG if retriever else SYSTEM
    user = task["prompt"]
    if retriever:
        user = f"# 参考条文\n{build_context(task, retriever, articles or [])}\n\n# 質問\n{task['prompt']}"

    rec = {"id": task["id"], "kind": task["kind"], "industry": task["industry"], "model": model}
    try:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        r = chat(model, msgs, max_tokens=3000)
    except Exception as e:
        rec.update({"correct": False, "error": str(e)[:300], "latency_s": None})
        return rec

    ok, why = grade(task, r["text"])
    u = r["usage"] or {}
    pin = u.get("prompt_tokens", 0)
    pout = u.get("completion_tokens", 0)
    p = pricing.get(model, {})
    cost = (pin * p.get("in", 0) + pout * p.get("out", 0)) / 1_000_000 if p else None

    rec.update({"correct": ok, "why": why, "pred": r["text"][:600],
                "answer": task["answer"], "latency_s": round(r["latency_s"], 3),
                "prompt_tokens": pin, "completion_tokens": pout, "cost_usd": cost,
                "finish": r["finish"]})
    return rec


def summarize(rows: list[dict], model: str) -> dict:
    ok = [r for r in rows if r.get("correct")]
    err = [r for r in rows if r.get("error")]
    lat = [r["latency_s"] for r in rows if r.get("latency_s")]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    by_kind: dict[str, dict] = {}
    by_ind: dict[str, dict] = {}
    for r in rows:
        for bucket, key in ((by_kind, r["kind"]), (by_ind, r["industry"])):
            d = bucket.setdefault(key, {"n": 0, "ok": 0})
            d["n"] += 1
            d["ok"] += 1 if r.get("correct") else 0
    return {
        "model": model,
        "n": len(rows),
        "accuracy": round(len(ok) / len(rows), 4) if rows else 0,
        "accuracy_answered": round(len(ok) / max(1, len(rows) - len(err)), 4),
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
    ap.add_argument("--industry", default="", help="tax/labor/finance（空=全業種）")
    ap.add_argument("--all", action="store_true", help="全業種")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kind", default="")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default=TASKS)
    a = ap.parse_args()

    if not KEY:
        print("TEAI_API_KEY が未設定です", file=sys.stderr)
        return 2

    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    if a.industry and not a.all:
        tasks = [t for t in tasks if t["industry"] == a.industry]
    if a.kind:
        tasks = [t for t in tasks if t["kind"] == a.kind]
    if a.limit:
        import random
        rng = random.Random(a.seed)
        tasks = rng.sample(tasks, min(a.limit, len(tasks)))

    pricing = load_pricing()
    # RAG は業種を混ぜない。全業種実行時はタスクの業種ごとに該当条文だけ引く。
    all_articles = load_articles()
    industries = sorted({t["industry"] for t in tasks})
    retrievers: dict[str, Retriever] = {}
    if a.rag:
        for ind in industries:
            ind_arts = [x for x in all_articles if x.get("industry") == ind]
            retrievers[ind] = Retriever(ind_arts)
    tag = a.tag or (("rag" if a.rag else "base") + ("-" + a.industry if a.industry else ""))
    outdir = os.path.join(ROOT, "results", tag)
    os.makedirs(outdir, exist_ok=True)

    summaries = []
    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        t0 = time.time()
        rows: list[dict] = []
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
