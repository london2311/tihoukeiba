# ===== nar/train.py     (学習・検証の実行スクリプト) =====
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
from . import edge as EDGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None,
                    help="学習/検証の境界日。既定はオッズ完備の開始日")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min-starts", type=int, default=3,
                    help="過去N走未満の馬を学習から除く")
    ap.add_argument("--sweep", action="store_true",
                    help="正則化と特徴集合を総当たりして比較表を出す")
    ap.add_argument("--temp", action="store_true",
                    help="温度較正を有効にする(既定オフ)。"
                         "学習末尾で当てたTは検証期で外れることが実測されている")
    ap.add_argument("--exact", action="store_true",
                    help="厳密な条件付きロジットを使う(既定は従来の近似)")
    ap.add_argument("--with-odds", action="store_true",
                    help="実オッズを特徴に入れ、オッズ完備レースだけで学習・検証する。"
                         "実運用に最も近い構成")
    ap.add_argument("--with-market", action="store_true",
                    help="市場の人気順位を特徴に入れる。"
                         "『市場に勝つ』のではなく『市場を補正する』方針")
    a = ap.parse_args()

    # --- 貼り間違い検知 ---
    #   ブラウザUIで作業しているとファイルの中身を取り違えることがある。
    #   実際に features.py / model.py が train.py の複製になった事故があった。
    #   分かりにくい AttributeError で落ちる前に、ここで止める。
    import sys
    for mod, attr, need in [(FT, "BASE_COLS", "nar/features.py"),
                            (FT, "add_speed_figure", "nar/features.py"),
                            (M, "ConditionalLogit", "nar/model.py")]:
        if not hasattr(mod, attr):
            sys.exit(f"!! {need} の中身が想定と違う ({attr} が無い)。\n"
                     f"   ファイルの貼り間違いを確認すること。\n"
                     f"   各ファイルの1行目に '# ===== nar/xxx.py' のマーカーがある。")
    if not hasattr(M.ConditionalLogit(), "temperature"):
        sys.exit("!! nar/model.py が古い (温度較正が入っていない)")

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

    # ★min_starts は学習だけに掛ける。検証では絶対に出走頭を間引かない★
    #   間引くとレース内 softmax が残った馬だけで1に正規化される。
    #   本当の勝ち馬が間引かれていた場合、予測確率だけが水増しされ、
    #   キャリブレーションもEVも実態より良く(高く)見える。
    #   2026-08-15 の実測では 平均予測 0.118 / 平均実現 0.080 と
    #   約1.5倍ずれていた。分割の後で学習側にだけ適用する。

    # -----------------------------------------------------------------
    # 市場特徴
    #   ★方針転換の選択肢★
    #     独立モデルを作って市場と勝負するのが従来方針だったが、
    #     実測でモデルの順位付けは市場に劣る(AUC 0.764 vs 0.816)。
    #     EVに必要なのは「市場と独立であること」ではなく
    #     「市場より正確であること」。人気順位は発走前に確定する
    #     公開情報で、100%埋まっている。これを土台に置いて
    #     自前の特徴で補正する方が、実務的には筋がよい。
    #   ※ final_popularity は締切時点の確定値。締切直前に賭ける前提なら
    #     使用は妥当だが、それより早く賭けるならズレる点に注意。
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # 実オッズ構成
    #   人気順位は「1.2倍の1番人気」と「3.0倍の1番人気」を区別できない。
    #   実測ではその差が logloss 0.0070 分あり、自前特徴の上積み
    #   0.0025 より大きい。実運用では発走前にオッズが見えるのだから、
    #   これを土台に置くのが本来の姿。
    #   ただしオッズは3,412レースにしか無いので、その中で切り直す。
    # -----------------------------------------------------------------
    ODDS_COLS = ["odds_logprob", "overround"]
    if a.with_odds:
        cov = (df.groupby("race_id")["final_win_odds"]
                 .transform(lambda s: s.notna().mean()))
        before_r = df["race_id"].nunique()
        df = df[cov > 0.99].copy()
        print(f"\n■ オッズ完備レースに限定: "
              f"{before_r:,} -> {df['race_id'].nunique():,}レース "
              f"/ {len(df):,}出走")
        if df.empty:
            print("!! オッズ完備レースが無い")
            return
        df["odds_logprob"] = np.log(
            np.clip(M.market_prob_from_odds(df), 1e-6, 1.0))

    MARKET_COLS = ["mkt_rank", "mkt_logprob"]
    df["mkt_rank"] = df["final_popularity"].astype(float)
    df["mkt_logprob"] = np.log(
        np.clip(M.market_prob_from_popularity(df), 1e-6, 1.0))

    # -----------------------------------------------------------------
    # ★モデルの前に、市場そのものの効率性を測る★
    #   どの区分でも回収率が控除率なりに沈んでいるなら、
    #   その券種で勝つ前提自体を見直す必要がある。
    #   人気順位は100%埋まっているので全レースで測れる。
    # -----------------------------------------------------------------
    if not a.sweep:
        pmap0 = M.load_payout_map(con)
        print("\n" + "=" * 66)
        print("■ 市場効率マップ (全レース。N番人気を毎回買う戦略)")
        me = M.market_efficiency(df, pmap0)
        for bt, g in me.groupby("券種"):
            print(f"\n  [{bt}]")
            print(g.drop(columns="券種").to_string(
                index=False, float_format=lambda v: f"{v:.1f}"))
        best = me.sort_values("回収率%", ascending=False).head(3)
        print("\n  回収率上位:")
        print(best.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        if (me["95%下限"] > 100).any():
            print("  ★下限が100%を超える区分がある。モデル無しで妙味あり★")
        else:
            print("  どの人気順位も回収率100%に届かない"
                  "(控除率18.8%を考えれば正常)")

        print("\n" + "=" * 66)
        print("■ 組み合わせ券種の効率性 (全レース。人気N-M-Lを毎回1点買う)")
        ce = M.combination_efficiency(df, pmap0)
        print(ce.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        hot = ce[ce["95%下限"] > 100]
        if len(hot):
            print("\n  ★下限が100%超の組み合わせ★")
            print(hot.to_string(index=False,
                                float_format=lambda v: f"{v:.1f}"))
        else:
            near = ce[ce["回収率%"] > 90]
            if len(near):
                print("\n  下限100%超は無いが、回収率90%超は存在する:")
                print(near.to_string(index=False,
                                     float_format=lambda v: f"{v:.1f}"))

        EDGE.report(con, df, pmap0)

        for key in ("n_starters", "baba_code", "class_rank"):
            if key not in df:
                continue
            t = M.market_efficiency_by(df, pmap0, key)
            if len(t):
                print(f"\n  [1番人気・単勝] {key} 別 (上位5/下位3)")
                print(pd.concat([t.head(5), t.tail(3)]).to_string(
                    index=False, float_format=lambda v: f"{v:.1f}"))

    # ---- 分割 ----
    odds_ok = df[df["final_win_odds"].notna()]
    if len(odds_ok) == 0:
        print("!! オッズがある行が無い。バックテストできない")
        return
    if a.cutoff:
        cutoff = a.cutoff
    elif a.with_odds:
        # オッズ完備区間の中で日付分割する(既定は後ろ40%を検証)
        dts = df["race_date"].drop_duplicates().sort_values()
        cutoff = str(dts.iloc[int(len(dts) * 0.6)])[:10]
    else:
        cutoff = str(odds_ok["race_date"].min())[:10]
    tr, te = M.time_split(df, cutoff)
    print(f"\n■ 時系列分割 (cutoff={cutoff})")
    print(f"  学習 {len(tr):,}出走 / {tr['race_id'].nunique():,}レース"
          f"  ({str(tr['race_date'].min())[:10]} .. {str(tr['race_date'].max())[:10]})")
    print(f"  検証 {len(te):,}出走 / {te['race_id'].nunique():,}レース"
          f"  ({str(te['race_date'].min())[:10]} .. {str(te['race_date'].max())[:10]})")

    allc = [c for c in FT.BASE_COLS + FT.RANK_COLS + MARKET_COLS + ODDS_COLS
            if c in df.columns]
    cols = [c for c in FT.FEATURE_COLS if c in df.columns]
    if a.with_market:
        cols = cols + MARKET_COLS
    if a.with_odds:
        cols = cols + [c for c in ODDS_COLS if c in df.columns]
    before = len(tr)
    tr = tr[tr["career_starts"] >= a.min_starts]
    print(f"  学習のみ過去{a.min_starts}走以上に絞る: {before:,} -> {len(tr):,}"
          f"   (検証は {len(te):,} 出走を全頭のまま使う)")
    tr = tr.dropna(subset=["finish_pos"])
    te = te.dropna(subset=["finish_pos"])
    for d in (tr, te):
        d[allc] = d[allc].fillna(0.0)

    y_te0 = (te["finish_pos"] == 1).astype(int).values
    p_pop0 = M.market_prob_from_popularity(te)
    sub0 = te[te["final_win_odds"].notna()].copy()
    idx0 = te["final_win_odds"].notna().values
    ll_mkt = np.nan
    if len(sub0):
        ll_mkt = M.evaluate("市場", y_te0[idx0],
                            M.market_prob_from_odds(sub0),
                            sub0["race_id"])["logloss"]

    # -----------------------------------------------------------------
    # 総当たり: 正則化 x 特徴集合
    #   ★順位特徴を入れるべきか、正則化をどこまで効かせるかは
    #     理屈では決まらない。1回の実行で全部測って選ぶ★
    # -----------------------------------------------------------------
    if a.sweep:
        base = [c for c in FT.BASE_COLS if c in df.columns]
        both = [c for c in FT.BASE_COLS + FT.RANK_COLS if c in df.columns]
        rows = []
        variants = [("近似", M.ConditionalLogit), ("厳密", M.ConditionalLogitExact)]
        sets = [("自前のみ", base), ("自前+順位", both),
                ("自前+市場", base + MARKET_COLS),
                ("市場のみ", MARKET_COLS)]
        if a.with_odds:
            oc = [c for c in ODDS_COLS if c in df.columns]
            sets = [("自前+オッズ", base + oc), ("オッズのみ", oc),
                    ("自前+市場+オッズ", base + MARKET_COLS + oc),
                    ("自前のみ", base)]
        for mname, klass in variants:
            for label, cc in sets:
                for C in (0.1, 1.0):
                    m = klass(C=C).fit(
                        tr, cc, calib_frac=0.15 if a.temp else 0.0)
                    p0 = m.predict_proba(te, raw=True)
                    e0 = M.evaluate("x", y_te0, p0, te["race_id"])
                    c0 = M.calibration_table(y_te0, p0)
                    row = {
                        "模型": mname,
                        "特徴": f"{label}({len(cc)})", "C": C,
                        "logloss": e0["logloss"], "auc": e0["auc"],
                        "top1%": e0["top1_hit"] * 100,
                        "較正乖離": c0["diff"].abs().max(),
                        "vs市場": e0["logloss"] - ll_mkt,
                    }
                    if m.temperature != 1.0:
                        p1 = m.predict_proba(te)
                        e1 = M.evaluate("x", y_te0, p1, te["race_id"])
                        c1 = M.calibration_table(y_te0, p1)
                        row["T"] = m.temperature
                        row["T後logloss"] = e1["logloss"]
                        row["T後乖離"] = c1["diff"].abs().max()
                    rows.append(row)
        r = pd.DataFrame(rows).sort_values("logloss")
        print("\n■ 総当たり (logloss昇順。vs市場が負ならモデルの勝ち)")
        print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        best = r.iloc[0]
        print(f"\n  最良: {best['模型']} / {best['特徴']} C={best['C']}  "
              f"logloss {best['logloss']:.4f}  (市場 {ll_mkt:.4f})")
        return

    # ---- 学習 ----
    print(f"\n■ 学習 ({len(cols)}特徴)")
    klass = M.ConditionalLogitExact if a.exact else M.ConditionalLogit
    print(f"  模型: {'厳密な条件付きロジット' if a.exact else '近似(レース内z化+二値)'}")
    m = klass(C=a.C).fit(tr, cols, calib_frac=0.15 if a.temp else 0.0)
    print(m.coef_table().head(12).to_string(index=False))
    if m.calib_cut:
        print(f"\n  温度較正 T={m.temperature:.3f} "
              f"(学習内の {m.calib_cut} 以降で推定)")
        print("    T>1 は予測が尖りすぎていたことを意味する")

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

    # 健全性: レース内softmaxなら 平均予測 == 平均実現 になるはず。
    # ずれていたら出走頭を間引いている(=確率の水増し)。
    mp, ma = p_model.mean(), y_te.mean()
    print(f"\n  平均予測 {mp:.4f} / 平均実現 {ma:.4f}  比 {mp/max(ma,1e-9):.3f}")
    if abs(mp / max(ma, 1e-9) - 1) > 0.03:
        print("    !! 確率が水増しされている。出走頭の間引きを疑うこと")

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
    if m.temperature != 1.0:
        ct0 = M.calibration_table(y_te, m.predict_proba(te, raw=True))
        print(f"  較正前の最大乖離 {ct0['diff'].abs().max():.4f}"
              f"  → 較正後 {M.calibration_table(y_te, p_model)['diff'].abs().max():.4f}")
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

    # -----------------------------------------------------------------
    # 券種別バックテスト (払戻テーブル使用。事前オッズ不要)
    #   ★単勝は市場が最も効率的。他券種に妙味が無いかを一次判定する★
    #   同時に「人気1位を買うだけ」の戦略と必ず比較する。
    #   モデルがそれに勝てないなら、モデルには価値が無い。
    # -----------------------------------------------------------------
    print("\n■ 券種別バックテスト (検証期・全レース。常に1点買い)")
    pmap = M.load_payout_map(con)
    print(f"  払戻データ {len(pmap):,}件")

    bt_model = M.backtest_payouts(te, p_model, pmap, label="モデル")

    # 市場ベースライン: 人気順そのまま
    p_mkt = M.market_prob_from_popularity(te)
    bt_mkt = M.backtest_payouts(te, p_mkt, pmap, label="人気順")

    both = pd.concat([bt_model, bt_mkt]).sort_values(["券種", "戦略"])
    print(both.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    piv = both.pivot(index="券種", columns="戦略", values="回収率%")
    if {"モデル", "人気順"}.issubset(piv.columns):
        piv["差"] = piv["モデル"] - piv["人気順"]
        print("\n  回収率の差 (モデル - 人気順。正ならモデルに上積みがある)")
        print(piv.sort_values("差", ascending=False)
                 .to_string(float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
