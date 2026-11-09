"""oracle 検証。問題生成・採点が決定論的で、正解表現を必ず正解と判定することを確認する。

- 各タスクの answer を「正解の言い方」として grade に通し、全問正解になること。
- わざと間違えた回答を grade に通し、不正解になること（採点が緩すぎないこと）。
- 生成を2回走らせて同一シードで同一結果になること（決定性）。
"""
from __future__ import annotations

import json
import sys

from bench import grade as G
from bench import tasks as T


def oracle_answer(t: dict) -> str:
    """正解を、実務で人が書きそうな自然文で表現する（採点の取りこぼし検出用）。"""
    kind = t["kind"]
    a = t["answer"]
    if kind == "tax_consumption":
        return f"消費税額は{a:,}円です（円未満切り捨て）。"
    if kind in ("tax_income", "tax_corporate"):
        return f"法人税額は{a:,}円です。" if kind == "tax_corporate" else f"所得税額は{a:,}円です。"
    if kind == "labor_overtime":
        return f"割増賃金の総額は{a:,}円です。"
    if kind == "labor_leave":
        return f"年次有給休暇は{a}日付与されます。"
    if kind == "labor_avgwage":
        return f"平均賃金は{a:,}円です。"
    if kind == "finance_ratecap":
        return f"上限金利は年{a}%です。"
    if kind == "finance_excess":
        return f"無効となる超過額は{a:,}円です。"
    if kind == "finance_penalty":
        return f"賠償額の上限は{a:,}円です。"
    return str(a)


def wrong_answer(t: dict) -> str:
    kind = t["kind"]
    a = t["answer"]
    if kind == "labor_leave":
        return f"年次有給休暇は{a + 3}日です。"
    if isinstance(a, float):
        return f"{a + 9999:.2f}"
    return f"{int(a) + 9999:,}円"


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tasks = T.generate_all(per_kind=per)
    fails = 0
    for t in tasks:
        ok, why = G.grade(t, oracle_answer(t))
        if not ok:
            fails += 1
            print(f"ORACLE-FAIL {t['id']} {t['kind']}: {why}")
    # 不正解検出
    for t in tasks:
        ok, _ = G.grade(t, wrong_answer(t))
        if ok:
            fails += 1
            print(f"WRONG-ACCEPTED {t['id']} {t['kind']}: {wrong_answer(t)}")
    # 決定性
    again = T.generate_all(per_kind=per)
    if json.dumps(tasks, ensure_ascii=False, sort_keys=True) != json.dumps(again, ensure_ascii=False, sort_keys=True):
        fails += 1
        print("NONDETERMINISTIC: 同一シードで生成結果が変わった")

    by_ind: dict[str, int] = {}
    for t in tasks:
        by_ind[t["industry"]] = by_ind.get(t["industry"], 0) + 1
    print(f"tasks={len(tasks)} by_industry={by_ind} fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
