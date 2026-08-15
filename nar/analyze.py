"""
収集済みDBの現状把握と、較正の検証。

    python -m nar.analyze --report      # データ概況
    python -m nar.analyze --calibrate   # 人気→オッズ較正 + 検証
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .core import connect
from . import oddscal


def data_report(con) -> None:
    print("=" * 64)
    print("■ データ概況")
    q = lambda s: con.execute(s).fetchone()[0]
    nr = q("SELECT COUNT(*) FROM races")
    ne = q("SELECT COUNT(*) FROM entries")
    print(f"  レース      {nr:>9,}")
    print(f"  出走        {ne:>9,}")
    print(f"  払戻        {q('SELECT COUNT(*) FROM payouts'):>9,}")

    r = con.execute("SELECT MIN(race_date),MAX(race_date) FROM races").fetchone()
    print(f"  期間        {r[0]} .. {r[1]}")

    print("\n■ 欠損状況 (出走ベース)")
    for col, label in [("final_win_odds", "単勝オッズ"),
                       ("final_popularity", "人気順位"),
                       ("finish_pos", "着順"),
                       ("time_sec", "タイム"),
                       ("last_3f", "上がり3F")]:
        n = q(f"SELECT COUNT(*) FROM results WHERE {col} IS NOT NULL")
        print(f"  {label:12s} {n:>9,}  ({n/ne*100:5.1f}%)")
    nh = q("SELECT COUNT(*) FROM entries WHERE horse_id IS NOT NULL")
    print(f"  {'血統番号':12s} {nh:>9,}  ({nh/ne*100:5.1f}%)")

    print("\n■ オッズが揃っている期間")
    rows = con.execute("""
        SELECT substr(r.race_date,1,7) ym,
               COUNT(*) n,
               SUM(CASE WHEN res.final_win_odds IS NOT NULL THEN 1 ELSE 0 END) o
        FROM races r JOIN results res USING(race_id)
        GROUP BY ym ORDER BY ym DESC LIMIT 18
    """).fetchall()
    for ym, n, o in rows:
        bar = "#" * int(o / n * 40) if n else ""
        print(f"  {ym}  {o:>7,}/{n:>7,}  {o/n*100:5.1f}% {bar}")

    print("\n■ 馬ごとの出走数 (過去走特徴が作れるか)")
    d = pd.read_sql_query("""
        SELECT horse_id, COUNT(*) n FROM entries
        WHERE horse_id IS NOT NULL GROUP BY horse_id
    """, con)
    if len(d):
        print(f"  ユニーク馬  {len(d):>9,}")
        print(f"  平均出走数  {d['n'].mean():>9.1f}")
        for k in (1, 2, 3, 5, 10):
            c = (d["n"] >= k).sum()
            print(f"  {k}走以上     {c:>9,}  ({c/len(d)*100:5.1f}%)")


def calibrate(con) -> None:
    table = oddscal.run(con)

    df = pd.read_sql_query("""
        SELECT r.race_id, r.race_date, r.baba_code, r.baba_name,
               r.distance_m, r.n_starters,
               res.horse_no, res.final_popularity, res.final_win_odds,
               res.finish_pos
        FROM races r JOIN results res USING(race_id)
        WHERE res.final_popularity IS NOT NULL
    """, con)
    df = oddscal.impute_odds(df, table)
    oddscal.summarize_coverage(df)
    m = oddscal.validate_against_payout(con, df)

    # 人気順位ごとの精度(ここが実務上いちばん効く)
    print("\n=== 人気順位別の推定精度 (1着馬) ===")
    print(f"  {'人気':>4} {'n':>7} {'実オッズ中央':>12} {'推定中央':>10} {'|誤差|中央':>10}")
    for pop, g in m[m["odds_source"] != "actual"].groupby("final_popularity"):
        if len(g) < 30 or pop > 10:
            continue
        print(f"  {int(pop):>4} {len(g):>7,} "
              f"{g['odds_true'].median():>12.1f} "
              f"{g['odds_est'].median():>10.1f} "
              f"{g['rel_err'].abs().median()*100:>9.1f}%")

    # 市場の暗黙確率が妥当か = overround チェック
    print("\n=== overround (推定オッズから) ===")
    sub = df[df["odds_est"].notna()].copy()
    ov = sub.groupby("race_id")["odds_est"].apply(lambda s: (1 / s).sum())
    print(f"  中央 {ov.median():.3f} → 控除率 {(1-1/ov.median())*100:.1f}%")
    print(f"  25%tile {ov.quantile(.25):.3f} / 75%tile {ov.quantile(.75):.3f}")
    if not (1.15 < ov.median() < 1.40):
        print("  !! 異常。較正が歪んでいる可能性")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    con = connect()
    if a.report or not (a.report or a.calibrate):
        data_report(con)
    if a.calibrate:
        calibrate(con)


if __name__ == "__main__":
    main()
