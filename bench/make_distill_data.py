#!/usr/bin/env python3
"""Shitate 仕立て用の学習データを作る（テスト情報を一切使わない）。

方針（kenchiku-bench の make_distill_data.py を多業種へ一般化）:
  - 教師 = qwen/qwen3.8-max（API）。生徒 = ローカル Qwen3-8B-4bit。
  - テスト問題・その正解・その質問文は学習に使わない。
  - 読解Q&Aは「テストに出ない条文」だけから作る（漏洩ゼロ）。
  - 計算練習はテストと別シード・別パラメータで新規生成する。

出力: data/distill/train.jsonl / valid.jsonl（MLX LoRA 用 messages 形式）
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from bench import tasks as T  # noqa: E402
from bench.run import Retriever, build_context, load_articles  # noqa: E402
from bench.grade import grade  # noqa: E402

BASE = os.environ.get("TEAI_BASE_URL", "https://api.teai.io/v1")
KEY = os.environ.get("TEAI_API_KEY", "")
TEACHER = os.environ.get("TEACHER_MODEL", "qwen/qwen3.8-max")
OUTDIR = os.path.join(ROOT, "data", "distill")


def test_refs() -> set[tuple[str, str]]:
    refs = set()
    for line in open(os.path.join(ROOT, "data", "tasks.jsonl"), encoding="utf-8"):
        t = json.loads(line)
        a = t.get("audit") or {}
        if a.get("law") and a.get("article") is not None:
            refs.add((a["law"], str(a["article"])))
    return refs


def teacher(messages: list[dict], max_tokens: int = 1200, retries: int = 4) -> str:
    body = json.dumps({"model": TEACHER, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0.2}).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{BASE}/chat/completions", data=body,
                                     headers={"Authorization": f"Bearer {KEY}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.load(r)
            return (j["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"teacher failed: {last}")


def rec(question: str, answer: str) -> dict:
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]}


def build_reading(articles, skip, per_article, rng, workers=8) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pool = [a for a in articles if (a["law"], str(a["article"])) not in skip]
    rng.shuffle(pool)

    def one(a):
        text = a["text"].strip()
        if len(text) < 40:
            return []
        caption = (a.get("caption") or "").strip("（）")
        law, art = a["law"], a["article"]
        prompt = (
            f"次の法令条文を読み、内容を正確に理解するための一問一答を{per_article}組作ってください。\n"
            "各問は条文の事実だけに基づき、答えは条文に書かれた内容のみにしてください。\n"
            "出力は JSON 配列のみ。各要素は {\"q\": 質問, \"a\": 答え}。余計な説明は書かない。\n\n"
            f"【法令】{law} 第{art}条 {caption}\n【条文】\n{text}"
        )
        try:
            raw = teacher([{"role": "user", "content": prompt}], max_tokens=1400)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {law}{art}: {e}", file=sys.stderr)
            return []
        s = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else ""
        try:
            pairs = json.loads(s)
        except json.JSONDecodeError:
            return []
        ctx = f"【{law}第{art}条 {caption}】{text}"
        rows = []
        for p in pairs[:per_article]:
            q, ans = p.get("q", "").strip(), p.get("a", "").strip()
            if q and ans:
                rows.append(rec(f"# 参考条文\n{ctx}\n\n# 質問\n{q}", ans))
        return rows

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, a) for a in pool]
        for i, f in enumerate(as_completed(futs)):
            out.extend(f.result())
            if (i + 1) % 50 == 0:
                print(f"  reading {i+1}/{len(pool)} -> {len(out)} pairs", file=sys.stderr)
    return out


def calc_answer(t: dict) -> str:
    """計算過程を見せる学習用の答え。最終値だけでなく根拠と式を書く。

    短答だけを学ばせると、モデルが計算推論を失い税率を混同した（実測:
    仕立て後が base より悪化）。参考条文の該当箇所→適用→計算→答えの順で書く。
    """
    a = t["audit"]
    law, art = a["law"], a["article"]
    lines = [f"【根拠】{law}第{art}条"]
    if a.get("law2"):
        lines[0] += f"・{a['law2']}第{a['article2']}条"
    if a.get("note"):
        lines.append(f"【適用】{a['note']}")
    lines.append(f"【計算】{a['calc']}")
    lines.append(f"【答え】{t['answer']}")
    return "\n".join(lines)


def build_calc(rng, n_each, retriever, articles, test_prompts: set,
               use_teacher: bool = False, teacher_model: str = "") -> list[dict]:
    """テストと別シード・別パラメータで計算練習を作る。テスト問題は使わない。

    use_teacher=True のときは、モデルがテスト時に見るのと同じ RAG 文脈を
    教師に渡して解法を生成させ、最終値が既知の正解と一致した例だけ残す
    （検証済み蒸留）。テンプレ答えだけでは推論を学べず仕立てが悪化した（実測）。
    """
    out = []
    for ind, gens in T.GENERATORS.items():
        for gen in gens:
            made = 0
            attempts = 0
            while made < n_each and attempts < n_each * 30:
                attempts += 1
                t = gen(rng)
                if t["prompt"].strip() in test_prompts:
                    continue
                ctx = build_context(t, retriever, articles)
                user = f"# 参考条文\n{ctx}\n\n# 質問\n{t['prompt']}"
                if use_teacher:
                    ans = teacher([{"role": "user", "content": user}], max_tokens=900)
                    # 教師の最終値が既知の正解と一致するものだけ採用
                    ok, _ = grade(t, ans)
                    if not ok:
                        continue
                else:
                    ans = calc_answer(t)
                out.append(rec(user, ans))
                made += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--per-article", type=int, default=1)
    ap.add_argument("--calc-each", type=int, default=40)
    ap.add_argument("--max-articles", type=int, default=0, help="0=全条文")
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    ap.add_argument("--teacher", action="store_true",
                    help="計算例を教師モデルの検証済み解法から作る（推奨）")
    ap.add_argument("--teacher-model", default="google/gemini-3.5-flash")
    a = ap.parse_args()

    if not KEY:
        print("TEAI_API_KEY が未設定です", file=sys.stderr)
        return 1

    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(a.seed)
    articles = load_articles()
    skip = test_refs()
    print(f"corpus {len(articles)} 条文 / テスト除外 {len(skip)} 条文", file=sys.stderr)

    reading_src = [x for x in articles if (x["law"], str(x["article"])) not in skip]
    if a.max_articles:
        reading_src = reading_src[: a.max_articles]
    if a.per_article > 0:
        reading = build_reading(reading_src, skip, a.per_article, rng)
    else:
        reading = []
    print(f"reading pairs: {len(reading)}", file=sys.stderr)

    retriever = Retriever(articles)
    test_prompts = {json.loads(l)["prompt"].strip()
                    for l in open(os.path.join(ROOT, "data", "tasks.jsonl"), encoding="utf-8")}
    if a.teacher:
        global TEACHER
        TEACHER = a.teacher_model
    calc = build_calc(random.Random(a.seed + 1), a.calc_each, retriever, articles,
                      test_prompts, use_teacher=a.teacher, teacher_model=a.teacher_model)
    print(f"calc pairs: {len(calc)}", file=sys.stderr)

    rows = reading + calc
    rng.shuffle(rows)
    nvalid = max(1, int(len(rows) * a.valid_ratio))
    valid, train = rows[:nvalid], rows[nvalid:]
    for name, data in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(os.path.join(OUTDIR, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote train={len(train)} valid={len(valid)} -> {OUTDIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
