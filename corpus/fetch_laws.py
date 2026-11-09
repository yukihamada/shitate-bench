#!/usr/bin/env python3
"""e-Gov 法令API から業種別の法令を取得し、条文単位のJSONLにする。

kenchiku-bench の方式を多業種へ一般化したもの。正解データはすべてこの条文
テキストから機械的に作る（生成AIに法令を書かせない）。

使い方:
  python3 corpus/fetch_laws.py            # 全業種を取得して corpus/articles.jsonl を再生成
  python3 corpus/fetch_laws.py --check    # 既存ファイルの整合だけ確認
  python3 corpus/fetch_laws.py --industry tax   # 特定業種だけ

出力: corpus/articles.jsonl（law フィールドで業種を判別できるよう industry も持つ）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "articles.jsonl")

API = "https://elaws.e-gov.go.jp/api/1/lawdata/{lawid}"

# 業種 -> {法令名: (e-Gov法令ID, 抽出する条番号の集合 or None=全条文)}
# AIが既に実務で使われている領域を中心に選ぶ。
# 地方税法のように巨大な法令は、必要な条（地方消費税の税率）だけを抽出する。
INDUSTRIES: dict[str, dict[str, tuple[str, set[str] | None]]] = {
    # 税務・会計
    "tax": {
        "消費税法": ("363AC0000000108", None),
        "所得税法": ("340AC0000000033", None),
        "法人税法": ("340AC0000000034", None),
        # 地方消費税の税率（78分の22＝国税分の22/78）。消費税法29条の国税分と
        # 合わさって標準10%・軽減8%になる。この根拠がないと RAG で 6.24% と
        # 答えてしまう（実測）。
        "地方税法": ("325AC0000000226", {"72の83"}),
    },
    # 労務・人事
    "labor": {
        "労働基準法": ("322AC0000000049", None),
        "労働契約法": ("419AC0000000128", None),
    },
    # 金融（貸金・利息）
    "finance": {
        "利息制限法": ("329AC0000000100", None),
        "出資法": ("329AC0000000195", None),
    },
}


def fetch(law_id: str) -> bytes:
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{law_id}.xml")
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return open(path, "rb").read()
    req = urllib.request.Request(API.format(lawid=law_id),
                                 headers={"User-Agent": "shitate-bench/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) < 5000:
        raise RuntimeError(f"{law_id}: response too small ({len(data)} bytes)")
    with open(path, "wb") as f:
        f.write(data)
    return data


def text_of(el: ET.Element) -> str:
    """要素配下のテキストを、ルビを落として1行に正規化する。"""
    parts: list[str] = []

    def walk(e: ET.Element):
        if e.tag in ("Rt", "Rp"):  # ルビ本体は捨てる（読み仮名で検索が壊れるため）
            return
        if e.text:
            parts.append(e.text)
        for c in e:
            walk(c)
            if c.tail:
                parts.append(c.tail)

    walk(el)
    s = "".join(parts)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def article_num(raw: str) -> str:
    """'12_2' -> '12の2'（e-Govの枝番号表記を日本語表記へ）"""
    return raw.replace("_", "の")


def parse(law_name: str, law_id: str, data: bytes, keep: set[str] | None = None) -> list[dict]:
    """本則（MainProvision）だけを条文単位に分解する。

    附則（SupplProvision）にも <Article Num="1"> 等があり、そのまま拾うと
    本則の第一条を上書きしてしまうため、MainProvision 配下だけを歩く。
    keep が指定されたら、その条番号だけを残す（巨大法令の必要条抽出）。
    """
    root = ET.fromstring(data)
    main = root.find(".//MainProvision")
    if main is None:
        raise RuntimeError(f"{law_name}: MainProvision が見つからない")
    out = []
    ctx: dict[str, str] = {}

    def walk(el: ET.Element):
        for c in el:
            tag = c.tag
            if tag == "Chapter":
                t = c.find("ChapterTitle")
                ctx["chapter"] = text_of(t) if t is not None else ""
                ctx.pop("section", None)
                walk(c)
            elif tag == "Section":
                t = c.find("SectionTitle")
                ctx["section"] = text_of(t) if t is not None else ""
                walk(c)
            elif tag == "Article":
                num_raw = c.get("Num") or ""
                art = article_num(num_raw)
                if keep is not None and art not in keep:
                    continue
                cap = c.find("ArticleCaption")
                title = c.find("ArticleTitle")
                body = text_of(c)
                if body:
                    out.append({
                        "law": law_name,
                        "law_id": law_id,
                        "article": art,
                        "article_label": text_of(title) if title is not None else "",
                        "caption": text_of(cap) if cap is not None else "",
                        "chapter": ctx.get("chapter", ""),
                        "section": ctx.get("section", ""),
                        "text": body,
                    })
            elif tag in ("AppdxTable", "AppdxFig", "AppdxStyle"):
                continue
            else:
                walk(c)

    walk(main)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--industry", default="", help="特定業種だけ取得")
    a = ap.parse_args()

    if a.check:
        rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
        print(f"{len(rows)} articles, sha256={hashlib.sha256(open(OUT,'rb').read()).hexdigest()[:16]}")
        for ind, laws in INDUSTRIES.items():
            n = sum(1 for r in rows if r.get("industry") == ind)
            print(f"  {ind}: {n}")
        return 0

    rows = []
    for industry, laws in INDUSTRIES.items():
        if a.industry and industry != a.industry:
            continue
        for name, (lid, keep) in laws.items():
            data = fetch(lid)
            arts = parse(name, lid, data, keep)
            for x in arts:
                x["industry"] = industry
            print(f"[{industry}] {name}: {len(arts)} articles ({len(data)} bytes)", file=sys.stderr)
            rows.extend(arts)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
