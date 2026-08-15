"""
人気順位 → 単勝オッズ の較正。

背景:
  NAR公式はオッズを直近数か月しか保存していない。
  → 全出走の約2割でしかオッズが揃わない。
  一方で「全馬の人気順位」と「1着馬の確定オッズ(払戻から)」は全期間で取れる。

方針:
  オッズ完備のレース(約3.4万出走)を教師データにして
      P(単勝オッズ | 人気順位, 頭数, 場, 距離帯)
  を推定し、欠損レースに適用する。

  さらに払戻データから「1着馬の実オッズ」が全レースで1点ずつ観測できるので、
  それを使って較正のバイアスを検証・補正する。★これが精度の担保になる★

重要な注意:
  較正で埋めたオッズは推定値であり、実オッズではない。
  EVの絶対値を信用しすぎないこと。順位付けには十分使えるが、
  「EV105%だから買い」という判断には、実オッズが揃った期間で検証すること。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =====================================================================
# 距離帯 (場ごとのコース形態差を吸収するための粗いビン)
# =====================================================================
def dist_band(m: int | float | None) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "unk"
    m = int(m)
    if m <= 1000:
        return "s"      # 短距離
    if m <= 1400:
        return "m"
    if m <= 1800:
        return "l"
    return "xl"


def field_band(n: int | float | None) -> str:
    """頭数。オッズ分布は頭数に強く依存する。"""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "unk"
    n = int(n)
    if n <= 8:
        return "small"
    if n <= 11:
        return "mid"
    return "large"


# =====================================================================
# 学習
# =====================================================================
def build_odds_table(con) -> pd.DataFrame:
    """
    オッズが揃っているレースだけを取り出す。
    「揃っている」= そのレースの全出走馬に final_win_odds がある、を厳密に判定する。
    一部しか無いレースを混ぜると分布が歪む。
    """
    df = pd.read_sql_query("""
        SELECT r.race_id, r.baba_code, r.distance_m, r.n_starters,
               res.horse_no, res.final_popularity, res.final_win_odds,
               res.finish_pos
        FROM races r
        JOIN results res USING (race_id)
        WHERE res.final_popularity IS NOT NULL
    """, con)

    have = df.groupby("race_id")["final_win_odds"].transform(
        lambda s: s.notna().all())
    full = df[have].copy()
    full["dband"] = full["distance_m"].map(dist_band)
    full["fband"] = full["n_starters"].map(field_band)
    return full


def fit_pop_to_odds(full: pd.DataFrame,
                    min_samples: int = 30) -> dict:
    """
    (場, 距離帯, 頭数帯, 人気順位) -> log(オッズ) の中央値/分位点。

    階層フォールバックを持たせる:
        場+距離帯+頭数帯+人気  →  距離帯+頭数帯+人気  →  頭数帯+人気  →  人気
    サンプルが薄いキーは上位階層に落とす。地方は場ごとの差が大きいので
    可能な限り細かいキーを使いたいが、薄いところで過学習すると害になる。
    """
    full = full[full["final_win_odds"] > 0].copy()
    full["log_odds"] = np.log(full["final_win_odds"])

    levels = [
        ("L1", ["baba_code", "dband", "fband", "final_popularity"]),
        ("L2", ["dband", "fband", "final_popularity"]),
        ("L3", ["fband", "final_popularity"]),
        ("L4", ["final_popularity"]),
    ]

    table = {}
    for name, keys in levels:
        g = full.groupby(keys)["log_odds"]
        agg = g.agg(["median", "mean", "std", "count"])
        agg = agg[agg["count"] >= (min_samples if name != "L4" else 1)]
        table[name] = {"keys": keys, "data": agg}
        print(f"  {name} {'+'.join(keys):50s} {len(agg):>6,}キー")
    return table


def _lookup(table: dict, row) -> tuple[float, str]:
    for name in ("L1", "L2", "L3", "L4"):
        keys = table[name]["keys"]
        data = table[name]["data"]
        try:
            k = tuple(row[c] for c in keys)
            k = k[0] if len(k) == 1 else k
            v = data.loc[k, "median"]
            if not np.isnan(v):
                return float(np.exp(v)), name
        except (KeyError, TypeError):
            continue
    return np.nan, "miss"


def impute_odds(df: pd.DataFrame, table: dict) -> pd.DataFrame:
    """
    final_win_odds が欠損している行に推定オッズを入れる。
    実測値は絶対に上書きしない。est_source 列でどこから来たか分かるようにする。
    """
    df = df.copy()
    if "dband" not in df:
        df["dband"] = df["distance_m"].map(dist_band)
    if "fband" not in df:
        df["fband"] = df["n_starters"].map(field_band)

    need = df["final_win_odds"].isna() & df["final_popularity"].notna()
    est, src = [], []
    for _, row in df[need].iterrows():
        v, s = _lookup(table, row)
        est.append(v)
        src.append(s)

    df["odds_est"] = df["final_win_odds"]
    df.loc[need, "odds_est"] = est
    df["odds_source"] = np.where(df["final_win_odds"].notna(), "actual", "miss")
    df.loc[need, "odds_source"] = src
    return df


# =====================================================================
# 検証 — 払戻の実測1着オッズと突き合わせる
# =====================================================================
def validate_against_payout(con, df: pd.DataFrame) -> pd.DataFrame:
    """
    払戻の単勝は「1着馬の確定オッズ×100」。
    これは全期間で取れるので、推定値の当たり具合を全期間で検証できる。
    ★較正が信用できるかどうかは、この関数の出力だけで判断すること★
    """
    pay = pd.read_sql_query("""
        SELECT race_id, combination AS horse_no, payout_yen
        FROM payouts WHERE bet_type='win'
    """, con)
    pay["horse_no"] = pd.to_numeric(pay["horse_no"], errors="coerce")
    pay["odds_true"] = pay["payout_yen"] / 100.0

    m = df.merge(pay[["race_id", "horse_no", "odds_true"]],
                 on=["race_id", "horse_no"], how="inner")
    m = m[m["odds_est"].notna() & (m["odds_true"] > 0)].copy()

    m["log_err"] = np.log(m["odds_est"]) - np.log(m["odds_true"])
    m["rel_err"] = m["odds_est"] / m["odds_true"] - 1

    print("\n=== 較正の検証 (1着馬・全期間) ===")
    print(f"  対象 {len(m):,}レース")
    for src, g in m.groupby("odds_source"):
        bias = g["log_err"].mean()
        print(f"  {src:8s} n={len(g):>7,}  "
              f"中央比 {np.exp(g['log_err'].median()):.3f}  "
              f"平均logバイアス {bias:+.4f}  "
              f"RMSE(log) {np.sqrt((g['log_err']**2).mean()):.3f}  "
              f"|相対誤差|中央 {g['rel_err'].abs().median()*100:.1f}%")
    return m


def summarize_coverage(df: pd.DataFrame) -> None:
    print("\n=== オッズ充足状況 ===")
    vc = df["odds_source"].value_counts()
    tot = len(df)
    for k, v in vc.items():
        print(f"  {k:8s} {v:>8,}  ({v/tot*100:5.1f}%)")


# =====================================================================
def run(con) -> dict:
    print("=== 人気順→オッズ 較正 ===")
    full = build_odds_table(con)
    n_race = full["race_id"].nunique()
    print(f"  オッズ完備: {len(full):,}出走 / {n_race:,}レース")
    if n_race < 500:
        print("  !! 教師データが少なすぎる。較正の信頼性は低い")

    table = fit_pop_to_odds(full)
    return table
