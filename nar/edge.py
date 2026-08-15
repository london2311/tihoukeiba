# ===== nar/edge.py      (区分スキャン: 市場の歪みを探す) =====
"""
「どの区分なら回収率が100%を超えるか」を、多重検定に耐える形で探す。

★背景★
  2026-08-16 の実測で、全レース一律の回収率は
      複勝1番人気 89.1% / 単勝1番人気 80.0% / 三連複1-2-3人気 77.2%
  が上限だった。100%まで11ポイント足りない。
  モデルで埋めるのは失敗した(自前特徴を実オッズに足すと悪化する)。
  残る可能性は「特定の区分でだけ市場が歪んでいる」こと。

★仮説: 薄いプールほど歪む★
  パリミュチュエルでは各プールが独立で、地方の平日1Rの単勝プールは
  数十万円規模。大口1本でオッズが崩れる。
  プール規模そのものは取れていないが、代理変数は揃っている:
      race_no      … 1R と 11R でプール規模が全く違う。一度も切っていない
      baba_code    … 場ごとの規模差
      n_starters   … 頭数
      overround    … 1/オッズの合計。異常なレースを特定できる(3,412レース)

★多重検定の規律 — ここが本体★
  27パターン x 十数区分を切れば、偶然100%を超えるセルは必ず出る。
  だから:
    (1) 期間を前半/後半に割る。前半で見つけ、後半で確かめる
    (2) 検定したセル数を必ず印字し、偶然の期待個数と比べる
    (3) 後半で再現しないものは「見つからなかった」と扱う
  前半だけで100%超のセルを見つけて喜ぶのは、この計画で既に一度やった
  失敗(三連単1-2-3人気 93.5% → 全期間で 69.5%)の再演になる。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import UNORDERED, load_payout_map, _pop_map


# =====================================================================
# 買い目の払戻ベクトルを作る
# =====================================================================
def anchor_returns(df: pd.DataFrame, pmap: dict, bet: str,
                   pops: tuple[int, ...]) -> pd.DataFrame:
    """『毎レース人気 pops の組を1点買う』の払戻を、レース単位で返す。

    戻り: race_id, ret(円/100円), それとレース属性
    """
    pm = _pop_map(df)
    race_attr = (df.drop_duplicates("race_id")
                   .set_index("race_id"))

    rows = []
    for rid, m in pm.items():
        if not all(k in m for k in pops):
            continue
        sel = [m[k] for k in pops]
        if len(set(sel)) != len(sel):
            continue
        key = "-".join(sorted((str(x) for x in sel), key=int)) \
            if bet in UNORDERED else "-".join(str(x) for x in sel)
        rows.append((rid, float(pmap.get((rid, bet, key), 0))))

    r = pd.DataFrame(rows, columns=["race_id", "ret"])
    keep = [c for c in ["race_date", "baba_code", "race_no", "n_starters",
                        "class_rank", "distance_m", "overround"]
            if c in race_attr.columns]
    return r.join(race_attr[keep], on="race_id")


def _ci(x: np.ndarray, rng, n_boot: int = 400) -> tuple[float, float]:
    n = len(x)
    b = [x[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# =====================================================================
# 区分スキャン (前半で探し、後半で確かめる)
# =====================================================================
def scan(rdf: pd.DataFrame, keys: list[str], min_n: int = 300,
         holdout_frac: float = 0.4, seed: int = 0) -> pd.DataFrame:
    """区分ごとの回収率を、前半(探索)と後半(確認)に分けて出す。"""
    rng = np.random.default_rng(seed)
    rdf = rdf.sort_values("race_date")
    dts = rdf["race_date"].drop_duplicates().sort_values()
    cut = dts.iloc[int(len(dts) * (1 - holdout_frac))]
    a = rdf[rdf["race_date"] < cut]
    b = rdf[rdf["race_date"] >= cut]

    out = []
    for key in keys:
        if key not in rdf.columns:
            continue
        col = rdf[key]
        if key == "overround":
            col = pd.cut(rdf[key], [0, 1.15, 1.20, 1.23, 1.26, 1.30, 9],
                         labels=["~1.15", "1.15-1.20", "1.20-1.23",
                                 "1.23-1.26", "1.26-1.30", "1.30~"])
        elif key == "distance_m":
            col = pd.cut(rdf[key], [0, 1000, 1300, 1600, 1900, 9999],
                         labels=["~1000", "1000-1300", "1300-1600",
                                 "1600-1900", "1900~"])
        g = pd.Series(col.values, index=rdf.index)

        for v, idx in rdf.groupby(g.values, observed=True).groups.items():
            sub_a = a.loc[a.index.intersection(idx), "ret"].values
            sub_b = b.loc[b.index.intersection(idx), "ret"].values
            if len(sub_a) < min_n:
                continue
            lo, hi = _ci(sub_a, rng)
            out.append({
                "区分": key, "値": str(v),
                "前半n": len(sub_a), "前半回収%": sub_a.mean(),
                "下限": lo, "上限": hi,
                "後半n": len(sub_b),
                "後半回収%": sub_b.mean() if len(sub_b) else np.nan,
            })
    return pd.DataFrame(out).sort_values("前半回収%", ascending=False)


# =====================================================================
# 報告
# =====================================================================
ANCHORS = [
    ("place", (1,), "複勝1番人気"),
    ("place", (2,), "複勝2番人気"),
    ("place", (3,), "複勝3番人気"),
    ("win",   (1,), "単勝1番人気"),
    ("wide",  (1, 2), "ワイド1-2番人気"),
    ("trio",  (1, 2, 3), "三連複1-2-3番人気"),
]

SEG_KEYS = ["race_no", "baba_code", "n_starters", "class_rank",
            "distance_m", "overround"]


def report(con, df: pd.DataFrame, pmap: dict | None = None,
           min_n: int = 300) -> None:
    if pmap is None:
        pmap = load_payout_map(con)

    print("\n" + "=" * 70)
    print("■ 区分スキャン — 回収率100%超の区分は実在するか")
    print("  前半で探し、後半で確かめる。後半で再現しないものは無いものとする。")

    total_cells = 0
    survivors = []

    for bet, pops, label in ANCHORS:
        rdf = anchor_returns(df, pmap, bet, pops)
        if rdf.empty:
            continue
        base = rdf["ret"].mean()
        print(f"\n{'-' * 70}")
        print(f"■ {label}   全体 {base:.1f}%  (n={len(rdf):,})")

        t = scan(rdf, SEG_KEYS, min_n=min_n)
        if t.empty:
            print("  区分が小さすぎて測れない")
            continue
        total_cells += len(t)

        print(t.head(8).to_string(index=False,
                                  float_format=lambda v: f"{v:.1f}"))

        # 前半で下限100%超 かつ 後半でも100%超 のものだけ残す
        s = t[(t["下限"] > 100) & (t["後半回収%"] > 100)]
        for _, r in s.iterrows():
            survivors.append({"買い目": label, **r.to_dict()})

    print("\n" + "=" * 70)
    print(f"■ 判定   検定したセル数 {total_cells}")
    # 片側5%で偶然「下限>100」が出る個数の目安
    print(f"  偶然 下限100%超 が出る期待個数の目安: {total_cells * 0.025:.1f}")
    if survivors:
        print("\n  ★前半・後半の両方で100%を超えた区分★")
        print(pd.DataFrame(survivors).to_string(
            index=False, float_format=lambda v: f"{v:.1f}"))
        print("\n  ただし上の期待個数と見比べること。"
              "1〜2個なら偶然の範囲内である可能性が高い。")
    else:
        print("\n  前半・後半の両方で100%を超えた区分は無い。")
        print("  → 人気順位を軸にした固定戦略では、どの区分でも控除率を超えない。")
