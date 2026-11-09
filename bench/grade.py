"""採点ロジック。すべて決定論的で、モデルの自己申告に依存しない。

kenchiku-bench の採点基盤（数値・分数・条番号の正規化）を共通化し、
税務・労務・金融の業種別 kind を追加したもの。
"""
from __future__ import annotations

import re

KANJI = "〇一二三四五六七八九"
KD = {c: i for i, c in enumerate(KANJI)}


def kanji_num(s: str) -> int | None:
    """'六十'->60, '百'->100, '千二百'->1200, '十二'->12"""
    if not s:
        return None
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in KD:
            num = KD[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
        else:
            return None
    return total + section + num


def parse_float(text: str) -> float | None:
    if not text:
        return None
    t = text.replace("．", ".").replace("，", ",")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


KANJI_NUM_RE = r"[〇一二三四五六七八九十百千]+"


def parse_fraction(text: str) -> float | None:
    if not text:
        return None
    t = text.replace(" ", "").replace("　", "").replace("．", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / b if b else None
    m = re.search(r"(\d+)\s*分の\s*(\d+)", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a:
            return b / a
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        v = float(m.group(1))
        return v / 100 if v > 1.5 else v
    return None


def numeric_candidates(text: str) -> set[str]:
    """テキスト中の数値表現を answer_norm と同じキー空間に写す。"""
    out: set[str] = set()
    if not text:
        return out
    t = text.replace("　", " ").replace("．", ".")
    for m in re.finditer(rf"({KANJI_NUM_RE}分の{KANJI_NUM_RE}|\d+\s*分の\s*\d+|\d+\s*/\s*\d+)", t):
        f = parse_fraction(m.group(0))
        if f is not None:
            out.add(f"frac:{f:.6f}")
    for m in re.finditer(rf"{KANJI_NUM_RE}", t):
        n = kanji_num(m.group(0))
        if n is not None:
            out.add(f"num:{n}")
    for m in re.finditer(r"\d+(?:\.\d+)?", t):
        out.add(f"num:{float(m.group(0)):g}")
    return out


def norm_article(s: str) -> str:
    """'第56条の2' / '56条の2' / '第五十六条の二' -> '56の2'"""
    t = s.replace("第", "").replace("条", "").replace(" ", "").replace("　", "")
    t = t.replace("の二", "の2").replace("の三", "の3").replace("の四", "の4")
    m = re.fullmatch(r"([〇一二三四五六七八九十百千]+)(の\d+)?", t)
    if m:
        n = kanji_num(m.group(1))
        if n is not None:
            return f"{n}{m.group(2) or ''}"
    return t


ART_REF = re.compile(r"第?\s*([0-9〇一二三四五六七八九十百千]+(?:の[0-9一二三四五六七八九十]+)?)\s*条")


def extract_articles(p: str) -> set[str]:
    q = re.sub(r"第\s*([0-9〇一二三四五六七八九十百千]+)\s*条\s*の\s*([0-9一二三四五六七八九十]+)",
               r"第\1の\2条", p)
    return {norm_article(m.group(1)) for m in ART_REF.finditer(q)}


def _nums(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "").replace("，", ""))]


def _has_num(text: str, want: float, tol: float) -> bool:
    return any(abs(v - want) <= tol for v in _nums(text))


# ---------------------------------------------------------------------------
# 業種別採点
# ---------------------------------------------------------------------------

def grade(task: dict, pred: str) -> tuple[bool, str]:
    kind = task.get("kind")
    if pred is None:
        return False, "empty"
    p = pred.strip()

    # --- 共通: 数値1つを答えさせる問題 ---
    if kind == "numeric":
        want = task["answer_norm"]
        cands = numeric_candidates(p)
        return (want in cands), ("ok" if want in cands else f"want={want} got={sorted(cands)[:6]}")

    # --- 共通: 金額・計算結果（許容誤差つき） ---
    if kind == "amount":
        tol = float(task.get("tol", 1.0))
        want = float(task["answer"])
        ok = _has_num(p, want, tol)
        return ok, f"want={want} tol={tol} got={_nums(p)[:6]}"

    # --- 共通: 率（%） ---
    if kind == "rate":
        tol = float(task.get("tol", 0.05))
        want = float(task["answer"])
        pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[%％]", p.replace(",", ""))]
        pool = pcts if pcts else _nums(p)
        ok = any(abs(v - want) <= tol for v in pool)
        return ok, f"want={want}% tol={tol} got={pool[:6]}"

    # --- 共通: 条文引用（法令名＋条番号） ---
    if kind == "cite":
        want_law, want_art = task["answer_norm"].split("#")
        # 法令名を明示しているか（取り違え防止）。略称ゆれは部分一致で許容。
        law_ok = want_law in p or want_law.replace("法", "") in p
        arts = extract_articles(p)
        ok = law_ok and (want_art in arts)
        return ok, f"want={want_law}#{want_art} law_ok={law_ok} arts={sorted(arts)[:4]}"

    # --- 共通: 誤り指摘＋正しい値＋条番号 ---
    if kind == "false_claim":
        said_wrong = bool(re.search(r"誤り|誤って|不正確|正しくない|誤解|incorrect|wrong", p))
        _, want_law, want_art, want_num = task["answer_norm"].split("#")
        arts = extract_articles(p)
        m = re.search(r"(?:正しい数値|正しくは|正しい金額|正しい率|correct(?:ed)? value)(.{0,200})", p, re.S)
        scope = p
        if m:
            for frag in re.split(r"[。\n]", m.group(1)):
                if numeric_candidates(frag):
                    scope = frag
                    break
            else:
                scope = m.group(1)
        num_ok = want_num in numeric_candidates(scope)
        ok = said_wrong and num_ok and (want_art in arts)
        return ok, f"wrong={said_wrong} num={num_ok} art={want_art in arts}"

    # --- 税務: 消費税額（税抜価格×税率、端数処理） ---
    if kind == "tax_consumption":
        want = float(task["answer"])
        tol = float(task.get("tol", 1.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} got={_nums(p)[:6]}"

    # --- 税務: 所得税（速算表・超過累進） ---
    if kind == "tax_income":
        want = float(task["answer"])
        tol = float(task.get("tol", 100.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} tol={tol} got={_nums(p)[:6]}"

    # --- 税務: 法人税（中小軽減・800万境界） ---
    if kind == "tax_corporate":
        want = float(task["answer"])
        tol = float(task.get("tol", 100.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} tol={tol} got={_nums(p)[:6]}"

    # --- 労務: 割増賃金（時間外・深夜・休日・月60h超） ---
    if kind == "labor_overtime":
        want = float(task["answer"])
        tol = float(task.get("tol", 1.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} got={_nums(p)[:6]}"

    # --- 労務: 年次有給休暇の日数 ---
    if kind == "labor_leave":
        want = int(task["answer"])
        # 「20日」「二十日」どちらも読む。条番号の数字を拾わないよう条引用を除去。
        q = re.sub(r"第?[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十]+)?(?:第?[0-9一二三四五六七八九十]+項)?", " ", p)
        days = [int(x) for x in re.findall(r"(\d+)\s*(?:日|労働日)", q)]
        days += [kanji_num(m) for m in re.findall(rf"({KANJI_NUM_RE})\s*日", q) if kanji_num(m) is not None]
        ok = want in days
        return ok, f"want={want} got={days[:6]}"

    # --- 労務: 平均賃金 ---
    if kind == "labor_avgwage":
        want = float(task["answer"])
        tol = float(task.get("tol", 1.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} got={_nums(p)[:6]}"

    # --- 金融: 上限金利（利息制限法の元本別） ---
    if kind == "finance_ratecap":
        want = float(task["answer"])
        tol = float(task.get("tol", 0.05))
        pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[%％]", p.replace(",", ""))]
        pool = pcts if pcts else _nums(p)
        ok = any(abs(v - want) <= tol for v in pool)
        return ok, f"want={want}% got={pool[:6]}"

    # --- 金融: 上限超過利息（無効部分の額） ---
    if kind == "finance_excess":
        want = float(task["answer"])
        tol = float(task.get("tol", 1.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} got={_nums(p)[:6]}"

    # --- 金融: 賠償額予定の上限（1.46倍 / 業者2割） ---
    if kind == "finance_penalty":
        want = float(task["answer"])
        tol = float(task.get("tol", 1.0))
        ok = _has_num(p, want, tol)
        return ok, f"want={want} got={_nums(p)[:6]}"

    return False, f"unknown-kind:{kind}"
