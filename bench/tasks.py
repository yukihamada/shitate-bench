"""決定論的な問題生成。正解はすべて法令の条文・法定率から機械計算し、
各問題に監査可能な根拠（法令・条番号・計算式）を付ける。

生成AIに問題や正解を書かせない。値はここで計算し、selftest で oracle 全問正解を
確認する（bench/selftest.py）。
"""
from __future__ import annotations

import random
from typing import Callable


# ---------------------------------------------------------------------------
# 税務・会計
# ---------------------------------------------------------------------------

# 消費税法29条: 標準 7.8%（国税）、軽減 6.24%。地方消費税 2.2%/1.76% を
# 含めた総額税率は 10% / 8%。本ベンチは「税抜価格に課す消費税額（国税＋地方）」を
# 10%/8% の総額で問う（実務の請求書と同じ）。根拠は消費税法29条＋地方税法。
def tax_consumption(rng: random.Random) -> dict:
    kind_choice = rng.choice(["standard", "reduced"])
    price = rng.choice([980, 1280, 1980, 2980, 4800, 6800, 12800, 39800, 98000])
    rate = 10 if kind_choice == "standard" else 8
    # 消費税額は円未満切り捨て（実務の端数処理。インボイス制度の端数処理は
    # 一の請求書につき1回。ここでは単一明細なので明細単位で切り捨て）。
    raw = price * rate / 100
    tax = int(raw)  # 切り捨て
    total = price + tax
    label = "標準税率10%" if rate == 10 else "軽減税率8%"
    return {
        "industry": "tax",
        "kind": "tax_consumption",
        "prompt": (
            f"税抜価格 {price} 円の商品（{label}対象）について、消費税額（国税＋地方消費税の合計）"
            f"を円未満切り捨てで計算し、税込価格も示してください。"
        ),
        "answer": tax,
        "answer_norm": f"num:{tax}",
        "tol": 0.5,
        "audit": {
            "law": "消費税法",
            "article": "29",
            "law2": "地方税法",
            "article2": "72の83",
            "note": f"消費税法29条の国税分（標準7.8%、軽減6.24%）＋地方税法72条の83の地方消費税（国税分の22/78）＝標準10%、軽減8%",
            "calc": f"{price} × {rate}% = {raw} → 切捨て {tax}（税込 {total}）",
        },
    }


# 所得税法89条の速算表（超過累進）。課税総所得金額に対する税額（100円未満切捨ては
# 国税通則法119条だが、ここでは円単位の理論値で問う）。
INCOME_BRACKETS = [
    (1_950_000, 0.05, 0),
    (3_300_000, 0.10, 97_500),
    (6_950_000, 0.20, 427_500),
    (9_000_000, 0.23, 636_000),
    (18_000_000, 0.33, 1_536_000),
    (40_000_000, 0.40, 2_796_000),
    (float("inf"), 0.45, 4_796_000),
]


def income_tax(taxable: int) -> tuple[int, float, int]:
    for limit, rate, deduct in INCOME_BRACKETS:
        if taxable <= limit:
            tax = int(taxable * rate - deduct)
            return tax, rate, deduct
    raise ValueError("unreachable")


def tax_income(rng: random.Random) -> dict:
    taxable = rng.choice([1_200_000, 1_500_000, 1_950_000, 2_400_000, 3_000_000,
                          3_300_000, 4_500_000, 6_950_000, 7_000_000, 9_000_000,
                          9_500_000, 12_000_000, 18_000_000, 20_000_000, 45_000_000])
    tax, rate, deduct = income_tax(taxable)
    return {
        "industry": "tax",
        "kind": "tax_income",
        "prompt": (
            f"課税総所得金額が {taxable:,} 円のとき、所得税法89条の速算表により"
            f"所得税額を計算してください（円単位、控除額を引いた後の税額）。"
        ),
        "answer": tax,
        "answer_norm": f"num:{tax}",
        "tol": 1.0,
        "audit": {
            "law": "所得税法",
            "article": "89",
            "note": "所得税法89条の超過累進税率と速算控除額",
            "calc": f"{taxable:,} × {int(rate*100)}% − {deduct:,} = {tax:,}",
        },
    }


# 法人税法66条: 普通法人 23.2%。中小法人（資本金1億円以下）は年800万円以下の
# 所得部分に 19% の軽減税率（66条2項）。本ベンチは中小法人の軽減適用を問う。
def tax_corporate(rng: random.Random) -> dict:
    income = rng.choice([2_000_000, 3_000_000, 5_000_000, 7_000_000, 8_000_000,
                         9_000_000, 10_000_000, 12_000_000, 15_000_000, 20_000_000,
                         25_000_000, 40_000_000])
    small_band = min(income, 8_000_000)
    large_band = max(income - 8_000_000, 0)
    tax = int(small_band * 0.19 + large_band * 0.232)
    return {
        "industry": "tax",
        "kind": "tax_corporate",
        "prompt": (
            f"資本金1億円以下の中小法人の各事業年度の所得が {income:,} 円のとき、"
            f"法人税法66条により法人税額を計算してください（年800万円以下の部分は19%、"
            f"超過部分は23.2%）。"
        ),
        "answer": tax,
        "answer_norm": f"num:{tax}",
        "tol": 1.0,
        "audit": {
            "law": "法人税法",
            "article": "66",
            "note": "法人税法66条1項（23.2%）と2項（中小軽減19%、800万円境界）",
            "calc": f"{small_band:,}×19% + {large_band:,}×23.2% = {tax:,}",
        },
    }


# ---------------------------------------------------------------------------
# 労務・人事
# ---------------------------------------------------------------------------

# 労働基準法37条: 時間外25%（月60h超は50%）、深夜25%加算、休日35%（政令）。
# 割増率は労基法37条＋労働基準法施行規則19条・政令。ここでは法定の最低率で計算。
def labor_overtime(rng: random.Random) -> dict:
    hourly = rng.choice([1200, 1500, 1800, 2000, 2500, 3000])
    hours = rng.choice([10, 20, 30, 45, 60, 70, 80])
    night = rng.choice([0, 0, 5, 10])
    # 月60h超の部分は50%、それ以外の時間外は25%。深夜は+25%。
    normal_ot = min(hours, 60)
    over60 = max(hours - 60, 0)
    # 深夜時間は時間外のうち night 時間とみなす（night<=hours を保証）
    night = min(night, hours)
    pay = normal_ot * hourly * 1.25 + over60 * hourly * 1.50 + night * hourly * 0.25
    pay = int(round(pay))
    return {
        "industry": "labor",
        "kind": "labor_overtime",
        "prompt": (
            f"時給 {hourly} 円の労働者が、1か月に時間外労働 {hours} 時間（うち深夜 {night} 時間）"
            f"行いました。労働基準法37条に基づき、これらの時間外・深夜労働に対して支払う"
            f"賃金の総額（基本給部分を含む。時間外は1.25倍、月60時間超は1.50倍、"
            f"深夜はさらに0.25倍を加算、休日労働なし）を計算してください。"
        ),
        "answer": pay,
        "answer_norm": f"num:{pay}",
        "tol": 1.0,
        "audit": {
            "law": "労働基準法",
            "article": "37",
            "note": "労基法37条1項（25%、60h超50%）・4項（深夜25%加算）",
            "calc": (f"{normal_ot}h×{hourly}×1.25 + {over60}h×{hourly}×1.50 "
                     f"+ {night}h×{hourly}×0.25 = {pay}"),
        },
    }


# 労働基準法39条: 6か月で10日、以降 1年+1, 2年+2, 3年+4, 4年+6, 5年+8, 6年以上+10。
# years は「雇入れからの継続勤務年数」。6か月経過日からの年数 = years - 0.5。
def leave_days(years: float) -> int:
    if years < 0.5:
        return 0
    since = round(years - 0.5, 1)  # 6か月経過日からの年数
    table = {0: 10, 1: 11, 2: 12, 3: 14, 4: 16, 5: 18}
    if since >= 6:
        return 20
    return table[int(since)]


def labor_leave(rng: random.Random) -> dict:
    # 継続勤務年数（6か月経過日から起算）を選ぶ
    years = rng.choice([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.0, 10.0, 15.0, 20.0])
    days = leave_days(years)
    return {
        "industry": "labor",
        "kind": "labor_leave",
        "prompt": (
            f"雇入れから {years} 年継続勤務し、全労働日の8割以上出勤した通常の労働者に"
            f"付与すべき年次有給休暇の日数を、労働基準法39条に基づき答えてください。"
        ),
        "answer": days,
        "answer_norm": f"num:{days}",
        "audit": {
            "law": "労働基準法",
            "article": "39",
            "note": "労基法39条1項・2項の付与日数表",
            "calc": f"継続勤務 {years} 年 → {days} 日",
        },
    }


# 労働基準法12条: 平均賃金 = 3か月の賃金総額 ÷ その期間の総日数。
def labor_avgwage(rng: random.Random) -> dict:
    monthly = rng.choice([200000, 250000, 300000, 350000, 400000])
    months = 3
    total = monthly * months
    days = rng.choice([89, 90, 91, 92])  # 3か月の総日数
    avg = total / days
    avg_r = round(avg, 2)
    return {
        "industry": "labor",
        "kind": "labor_avgwage",
        "prompt": (
            f"月給 {monthly:,} 円の労働者について、直近3か月の賃金総額が {total:,} 円、"
            f"その期間の総日数が {days} 日であるとき、労働基準法12条の平均賃金を"
            f"計算してください（円、小数第2位まで）。"
        ),
        "answer": avg_r,
        "answer_norm": f"num:{avg_r:g}",
        "tol": 1.0,
        "audit": {
            "law": "労働基準法",
            "article": "12",
            "note": "労基法12条1項（3か月賃金総額÷総日数）",
            "calc": f"{total:,} ÷ {days} = {avg_r}",
        },
    }


# ---------------------------------------------------------------------------
# 金融（貸金・利息）
# ---------------------------------------------------------------------------

# 利息制限法1条: 元本10万円未満=年20%、10万以上100万未満=18%、100万以上=15%。
def rate_cap(principal: int) -> int:
    if principal < 100_000:
        return 20
    if principal < 1_000_000:
        return 18
    return 15


def finance_ratecap(rng: random.Random) -> dict:
    principal = rng.choice([30_000, 50_000, 80_000, 99_999, 100_000, 200_000,
                            300_000, 500_000, 999_999, 1_000_000, 2_000_000,
                            3_000_000, 5_000_000, 10_000_000])
    cap = rate_cap(principal)
    return {
        "industry": "finance",
        "kind": "finance_ratecap",
        "prompt": (
            f"元本 {principal:,} 円の金銭消費貸借について、利息制限法1条により"
            f"利息の上限となる年利率（%）を答えてください。"
        ),
        "answer": cap,
        "answer_norm": f"num:{cap}",
        "tol": 0.05,
        "audit": {
            "law": "利息制限法",
            "article": "1",
            "note": "利息制限法1条の元本別上限金利",
            "calc": f"元本 {principal:,} 円 → 年 {cap}%",
        },
    }


# 利息制限法1条: 上限を超える利息は超過部分が無効。1年分の利息で超過額を計算。
def finance_excess(rng: random.Random) -> dict:
    principal = rng.choice([100_000, 500_000, 1_000_000, 2_000_000])
    cap = rate_cap(principal)
    charged = cap + rng.choice([2, 3, 5, 8])  # 上限超過の約定利率
    year = rng.choice([1, 1, 2])
    legal = principal * cap / 100 * year
    actual = principal * charged / 100 * year
    excess = int(round(actual - legal))
    return {
        "industry": "finance",
        "kind": "finance_excess",
        "prompt": (
            f"元本 {principal:,} 円を年 {charged}% で {year} 年間借りた場合、"
            f"利息制限法1条の上限（年{cap}%）を超える利息のうち、無効となる超過額は"
            f"いくらですか（円）。"
        ),
        "answer": excess,
        "answer_norm": f"num:{excess}",
        "tol": 1.0,
        "audit": {
            "law": "利息制限法",
            "article": "1",
            "note": "利息制限法1条（超過部分は無効）",
            "calc": (f"元本{principal:,}×({charged}%−{cap}%)×{year}年 = {excess:,}"),
        },
    }


# 利息制限法4条: 賠償額予定の上限は上限金利の1.46倍。業者（営業的貸金）は7条で年20%。
def finance_penalty(rng: random.Random) -> dict:
    principal = rng.choice([80_000, 100_000, 200_000, 500_000, 800_000,
                            1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000])
    cap = rate_cap(principal)
    limit_rate = cap * 1.46  # 4条
    penalty = int(round(principal * limit_rate / 100))
    return {
        "industry": "finance",
        "kind": "finance_penalty",
        "prompt": (
            f"元本 {principal:,} 円の金銭消費貸借で、債務不履行時の賠償額を予定する場合、"
            f"利息制限法4条により有効な賠償額の上限はいくらですか（円）。"
            f"（元本に応じた上限金利の1.46倍で計算）"
        ),
        "answer": penalty,
        "answer_norm": f"num:{penalty}",
        "tol": 1.0,
        "audit": {
            "law": "利息制限法",
            "article": "4",
            "note": "利息制限法4条1項（上限金利の1.46倍）",
            "calc": f"{principal:,}×{cap}%×1.46 = {penalty:,}",
        },
    }


GENERATORS: dict[str, list[Callable[[random.Random], dict]]] = {
    "tax": [tax_consumption, tax_income, tax_corporate],
    "labor": [labor_overtime, labor_leave, labor_avgwage],
    "finance": [finance_ratecap, finance_excess, finance_penalty],
}


def generate(industry: str, per_kind: int = 10, seed: int = 20261109) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    gid = 0
    for gen in GENERATORS[industry]:
        seen = set()
        made = 0
        attempts = 0
        while made < per_kind and attempts < per_kind * 50:
            attempts += 1
            t = gen(rng)
            key = (t["prompt"], t["answer"])
            if key in seen:
                continue
            seen.add(key)
            t["id"] = f"{industry}-{gid:03d}"
            gid += 1
            tasks.append(t)
            made += 1
    return tasks


def generate_all(per_kind: int = 10) -> list[dict]:
    out = []
    for ind in GENERATORS:
        out.extend(generate(ind, per_kind=per_kind))
    return out


if __name__ == "__main__":
    import json
    import sys
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tasks = generate_all(per_kind=per)
    for t in tasks:
        print(json.dumps(t, ensure_ascii=False))
