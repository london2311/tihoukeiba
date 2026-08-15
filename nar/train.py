"""
学習と検証の実行。

    python -m nar.train

流れ:
  1) 16万出走から特徴量を作る (features.py, リーク防止済み)
  2) 時系列分割で学習
  3) 市場ベースラインと比較        ← ここが実質の合否判定
  4) キャリブレーション確認
  5) オッズ完備レースだけで単勝EVバックテスト
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from .core import connect
from . import features as FT
from . import model as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None,
                    help="学習/検証の境界日。既定はオッズ完備の開始日")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min-starts", type=int, default=3,
                    help="過去N走未満の馬を学習から除く")
    a = ap.parse_args()

    con = connect()

    print("=" * 66)
    print("■ 特徴量生成")
    df = FT.build(con)
    FT.assert_no_leak(df)
    # ★レース内 z化で消える特徴を検出する★
    #   この模型はレース内で相対化するので、レース内定数は学習に寄与しない。
    #   「40特徴あるつもりで実は24」という事故を機械的に防ぐ。
    FT.assert_varies_within_race(df)
    FT.coverage_report(df)
    print(f"  {len(df):,}出走 / {df['race_id'].nunique():,}レース")

    # クラス復元の効き具合 (race_name から作り直したもの)
    if "class_rank" in df:
        ok = df["class_rank"].notna().mean() * 100
        mv = df["class_move"].notna().mean() * 100
        print(f"  class_rank 充足 {ok:5.1f}%  /  class_move 充足 {mv:5.1f}%")

    # 過去走が薄い馬は除く(特徴が全部NaNなのでノイズにしかならない)
    before = len(df)
    df = df[df["career_starts"] >= a.min_starts]
    print(f"  過去{a.min_starts}走以上に絞る: {before:,} -> {len(df):,}")

    # ---- 分割 ----
    odds_ok = df[df["final_win_odds"].notna()]
    if len(odds_ok) == 0:
        print("!! オッズがある行が無い。バックテストできない")
        return
    cutoff = a.cutoff or str(odds_ok["race_date"].min())[:10]
    tr, te = M.time_split(df, cutoff)
    print(f"\n■ 時系列分割 (cutoff={cutoff})")
    print(f"  学習 {len(tr):,}出走 / {tr['race_id'].nunique():,}レース"
          f"  ({str(tr['race_date'].min())[:10]} .. {str(tr['race_date'].max())[:10]})")
    print(f"  検証 {len(te):,}出走 / {te['race_id'].nunique():,}レース"
          f"  ({str(te['race_date'].min())[:10]} .. {str(te['race_date'].max())[:10]})")

    cols = [c for c in FT.FEATURE_COLS if c in df.columns]
    tr = tr.dropna(subset=["finish_pos"])
    te = te.dropna(subset=["finish_pos"])
    for d in (tr, te):
        d[cols] = d[cols].fillna(0.0)

    # ---- 学習 ----
    print(f"\n■ 学習 ({len(cols)}特徴)")
    m = M.ConditionalLogit(C=a.C).fit(tr, cols)
    print(m.coef_table().head(12).to_string(index=False))

    # ---- 評価 ----
    y_te = (te["finish_pos"] == 1).astype(int).values
    p_model = m.predict_proba(te)
    p_pop = M.market_prob_from_popularity(te)

    print("\n■ 検証 (全検証レース)")
    rows = [M.evaluate("model", y_te, p_model, te["race_id"]),
            M.evaluate("市場(人気順)", y_te, p_pop, te["race_id"])]

    sub = te[te["final_win_odds"].notna()].copy()
    if len(sub):
        idx = te["final_win_odds"].notna().values
        p_odds = M.market_prob_from_odds(sub)
        rows.append(M.evaluate("市場(実オッズ)", y_te[idx], p_odds,
                               sub["race_id"]))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ll_m = r.loc[r["name"] == "model", "logloss"].iloc[0]
    ll_k = r.loc[r["name"].str.contains("実オッズ"), "logloss"]
    print("\n  判定:")
    if len(ll_k) and ll_m < ll_k.iloc[0]:
        print(f"    モデルが市場を上回っている "
              f"(logloss {ll_m:.4f} < {ll_k.iloc[0]:.4f})")
    elif len(ll_k):
        print(f"    ★モデルは市場に負けている "
              f"(logloss {ll_m:.4f} >= {ll_k.iloc[0]:.4f})")
        print("     この状態でEVがプラスに見えても、それは偶然かリーク。")

    # ---- キャリブレーション ----
    print("\n■ キャリブレーション (予測確率 vs 実現率)")
    ct = M.calibration_table(y_te, p_model)
    print(ct.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    md = ct["diff"].abs().max()
    print(f"  最大乖離 {md:.4f}  "
          f"{'(良好)' if md < 0.03 else '(!! ズレが大きい。EVは信用できない)'}")

    # ---- EVバックテスト ----
    print("\n■ 単勝EVバックテスト (実オッズのみ)")
    if len(sub) == 0:
        print("  検証区間にオッズ完備レースが無い")
        return
    p_sub = m.predict_proba(sub)
    bt = M.backtest_win(sub, p_sub)
    bt["EV閾値"] = bt["EV閾値"].map(lambda v: f"{v:.2f}")
    print(bt.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n  ※ 回収率の点推定ではなく95%区間の下限を見ること。")
    print("     下限が100%を割るなら、それは『勝てる』とは言えない。")


if __name__ == "__main__":
    main()
