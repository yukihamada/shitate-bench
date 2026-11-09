# shitate-bench

AIが既に実務で使われている業種（税務会計・労務・金融）について、**正解が法令の条文と
法定の率・計算式から機械的に決まる**ベンチマーク。各問題には監査可能な根拠
（法令名・条番号・計算式）を付ける。生成AIに問題や正解を書かせない。

建築ベンチ [kenchiku-bench](../kenchiku-bench) の方式（条文取得 → 決定論的タスク生成 →
機械採点 → モデル実測 → 公開）を多業種へ一般化したもの。Shitate で各業種向けに
仕立てたモデルの前後比較に使う。

## 業種と法令（e-Gov法令APIから取得）

| 業種 | 法令 | e-Gov法令ID |
|---|---|---|
| tax（税務会計） | 消費税法 / 所得税法 / 法人税法 | 363AC0000000108 / 340AC0000000033 / 340AC0000000034 |
| labor（労務人事） | 労働基準法 / 労働契約法 | 322AC0000000049 / 419AC0000000128 |
| finance（金融） | 利息制限法 / 出資法 | 329AC0000000100 / 329AC0000000195 |

## タスク種別（すべて決定論的）

| kind | 内容 | 根拠 |
|---|---|---|
| tax_consumption | 消費税額（10%/8%、円未満切捨て） | 消費税法29条 |
| tax_income | 所得税額（超過累進・速算控除） | 所得税法89条 |
| tax_corporate | 法人税額（中小軽減19%、800万境界、23.2%） | 法人税法66条 |
| labor_overtime | 割増賃金（時間外25%、月60h超50%、深夜25%加算） | 労働基準法37条 |
| labor_leave | 年次有給休暇の付与日数 | 労働基準法39条 |
| labor_avgwage | 平均賃金（3か月賃金総額÷総日数） | 労働基準法12条 |
| finance_ratecap | 利息の上限金利（元本別 20/18/15%） | 利息制限法1条 |
| finance_excess | 上限超過利息の無効額 | 利息制限法1条 |
| finance_penalty | 賠償額予定の上限（上限金利×1.46） | 利息制限法4条 |

## 使い方

```bash
# 1. 法令コーパス取得（初回のみ。corpus/articles.jsonl を生成）
python3 corpus/fetch_laws.py

# 2. 問題生成（data/tasks.jsonl）
python3 bench/tasks.py 10          # 各kind 10問

# 3. oracle検証（正解表現が全問正解・誤答が全問不正解・決定性）
python3 -m bench.selftest 10

# 4. モデル実測（teai API）
export TEAI_API_KEY=...
python3 bench/run.py --models gpt-4.1-nano,claude-haiku-5-5 --all --tag base
python3 bench/run.py --models gpt-4.1-nano,claude-haiku-5-5 --all --rag --tag rag

# 5. レポート生成
python3 bench/report.py base rag
```

## 設計原則

- **正解は機械計算**：値は `bench/tasks.py` で法令の率から計算する。生成AIに書かせない。
- **監査可能性**：各タスクに `audit`（法令・条番号・計算式）を付ける。
- **採点は決定論的**：`bench/grade.py` はモデルの自己申告に依存せず、数値・率・条番号を
  正規化して照合する。oracle が全問正解・誤答が全問不正解になることを selftest で担保。
- **業種を混ぜない RAG**：税務モデルに労基法を渡さない。業種ごとに対象法令だけ引く。
