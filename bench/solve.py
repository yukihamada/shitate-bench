"""決定論ソルバ。モデルが抽出した「ケース条件」から、法令の率・式で金額を確定する。

モデルの仕事＝条文の選択とケース条件（価格・所得・時間数・元本など）の抽出。
計算の仕事＝ここ。算術ミスを構造的に排除するのが狙い。
tasks.py の純粋関数（income_tax / leave_days / rate_cap）を共有し、式の二重管理を避ける。
"""
from __future__ import annotations

from bench.tasks import income_tax, leave_days, rate_cap


def _i(x) -> int:
    return int(round(float(x)))


def solve_tax_consumption(s: dict) -> float:
    price = _i(s["price"])
    rate = _i(s["rate"])
    return int(price * rate / 100)


def solve_tax_income(s: dict) -> float:
    tax, _, _ = income_tax(_i(s["taxable"]))
    return tax


def solve_tax_corporate(s: dict) -> float:
    income = _i(s["income"])
    small = min(income, 8_000_000)
    large = max(income - 8_000_000, 0)
    tax = int(small * 0.15 + large * 0.232)
    return (tax // 100) * 100  # 100円未満切捨て（国税通則法119条）


def solve_labor_overtime(s: dict) -> float:
    hourly = _i(s["hourly"])
    hours = _i(s["hours"])
    night = min(_i(s.get("night", 0)), hours)
    normal = min(hours, 60)
    over60 = max(hours - 60, 0)
    pay = normal * hourly * 1.25 + over60 * hourly * 1.50 + night * hourly * 0.25
    return int(round(pay))


def solve_labor_leave(s: dict) -> float:
    return leave_days(float(s["years"]))


def solve_labor_avgwage(s: dict) -> float:
    total = _i(s["total"])
    days = _i(s["days"])
    return round(total / days, 2)


def solve_finance_ratecap(s: dict) -> float:
    return rate_cap(_i(s["principal"]))


def solve_finance_excess(s: dict) -> float:
    principal = _i(s["principal"])
    charged = float(s["charged"])
    year = _i(s["year"])
    cap = rate_cap(principal)
    return int(round(principal * (charged - cap) / 100 * year))


def solve_finance_penalty(s: dict) -> float:
    principal = _i(s["principal"])
    cap = rate_cap(principal)
    return int(round(principal * (cap * 1.46) / 100))


SOLVERS = {
    "tax_consumption": solve_tax_consumption,
    "tax_income": solve_tax_income,
    "tax_corporate": solve_tax_corporate,
    "labor_overtime": solve_labor_overtime,
    "labor_leave": solve_labor_leave,
    "labor_avgwage": solve_labor_avgwage,
    "finance_ratecap": solve_finance_ratecap,
    "finance_excess": solve_finance_excess,
    "finance_penalty": solve_finance_penalty,
}

# モデルに渡す抽出スキーマ（kind ごとの必要キー）。
SLOTS = {
    "tax_consumption": ["price", "rate"],
    "tax_income": ["taxable"],
    "tax_corporate": ["income"],
    "labor_overtime": ["hourly", "hours", "night"],
    "labor_leave": ["years"],
    "labor_avgwage": ["total", "days"],
    "finance_ratecap": ["principal"],
    "finance_excess": ["principal", "charged", "year"],
    "finance_penalty": ["principal"],
}

SLOT_DESC = {
    "price": "税抜価格(円)",
    "rate": "消費税率(10 または 8)",
    "taxable": "課税総所得金額(円)",
    "income": "法人の所得金額(円)",
    "hourly": "時給(円)",
    "hours": "時間外労働の合計時間",
    "night": "うち深夜労働の時間(なければ0)",
    "years": "雇入れからの継続勤務年数(小数可)",
    "total": "3か月の賃金総額(円)",
    "days": "その期間の総日数",
    "principal": "元本(円)",
    "charged": "約定年利率(%)",
    "year": "年数",
}


def solve(kind: str, slots: dict) -> float:
    return SOLVERS[kind](slots)
