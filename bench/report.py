#!/usr/bin/env python3
"""results/<tag>/summary.json から比較レポート（Markdown）を生成する。

使い方:
  python3 bench/report.py base            # 単一タグ
  python3 bench/report.py base rag        # 複数タグを横並び比較（base→ragの差も出す）
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IND_LABEL = {"tax": "税務会計", "labor": "労務人事", "finance": "金融"}
KIND_LABEL = {
    "tax_consumption": "消費税額", "tax_income": "所得税額", "tax_corporate": "法人税額",
    "labor_overtime": "割増賃金", "labor_leave": "有給休暇", "labor_avgwage": "平均賃金",
    "finance_ratecap": "上限金利", "finance_excess": "超過利息", "finance_penalty": "賠償額上限",
}


def load(tag: str) -> dict[str, dict]:
    p = os.path.join(ROOT, "results", tag, "summary.json")
    if not os.path.exists(p):
        return {}
    return {x["model"]: x for x in json.load(open(p))}


def table(tags: list[str]) -> str:
    data = {t: load(t) for t in tags}
    models = sorted({m for d in data.values() for m in d})
    lines = []
    if len(tags) == 1:
        t = tags[0]
        lines.append(f"## {t}\n")
        lines.append("| モデル | 総合 | 税務 | 労務 | 金融 | 誤答/エラー | 中央レイテンシ |")
        lines.append("|---|---|---|---|---|---|---|")
        for m in models:
            s = data[t][m]
            bi = s.get("by_industry", {})
            def a(k):
                return f"{bi.get(k,{}).get('accuracy','-')}"
            lines.append(f"| {m} | **{s['accuracy']:.1%}** | {a('tax')} | {a('labor')} | {a('finance')} "
                         f"| {s['errors']} | {s.get('latency_median_s','-')}s |")
    else:
        # 複数タグ（例: base と rag）の横並び
        head = " | ".join(tags)
        lines.append(f"## {' vs '.join(tags)}\n")
        lines.append(f"| モデル | {head} | 差(最後-最初) |")
        lines.append("|---|" + "---|" * len(tags) + "---|")
        for m in models:
            cells = []
            vals = []
            for t in tags:
                s = data[t].get(m)
                if s:
                    cells.append(f"{s['accuracy']:.1%}")
                    vals.append(s["accuracy"])
                else:
                    cells.append("-")
            diff = f"{(vals[-1]-vals[0])*100:+.1f}pt" if len(vals) >= 2 else "-"
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {diff} |")
    return "\n".join(lines)


def kind_table(tag: str) -> str:
    d = load(tag)
    if not d:
        return ""
    models = sorted(d)
    kinds = sorted({k for s in d.values() for k in s.get("by_kind", {})})
    lines = [f"\n### {tag}: 種別正答率\n"]
    lines.append("| 種別 | " + " | ".join(models) + " |")
    lines.append("|---|" + "---|" * len(models))
    for k in kinds:
        cells = []
        for m in models:
            v = d[m].get("by_kind", {}).get(k, {}).get("accuracy")
            cells.append(f"{v:.0%}" if v is not None else "-")
        lines.append(f"| {KIND_LABEL.get(k,k)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    tags = sys.argv[1:] or ["base"]
    out = ["# shitate-bench 結果\n",
           "正解は法令の条文・法定率から機械計算。採点は決定論的（oracle検証済み）。\n"]
    out.append(table(tags))
    for t in tags:
        out.append(kind_table(t))
    text = "\n".join(x for x in out if x) + "\n"
    path = os.path.join(ROOT, "REPORT.md")
    open(path, "w").write(text)
    print(text)
    print(f"\n-> {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
