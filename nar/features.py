"""
特徴量生成。

★このファイルで最も重要なのはリーク防止★
  「その馬の戦績」を使う以上、当該レース時点で知り得た情報しか使ってはいけない。
  ここを間違えると、バックテストで回収率150%が出て、実運用で70%に落ちる。
  競輪でも競艇でも、この事故が最も多い。

  対策:
    - 全ての集計に race_date < 当該race_date の厳格な条件を入れる
    - 同日開催の他レースも使わない(発走順が前でも使わない。取り切れないので)
    - 馬体重・単勝オッズは当日のものだが「レース前に確定」なので使用可
    - 走破タイム・上がり3F・通過順は当該レースのものは絶対に使わない
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np

# 当該レースの結果由来で、特徴量に混ぜてはいけない列
LEAK_COLS = {"finish_pos", "time_sec", "margin", "last_3f",
             "corner_1", "corner_2", "corner_3", "corner_4", "dnf_reason"}


def load_frame(con: sqlite3.Connection, min_date: str | None = None) -> pd.DataFrame:
    """entries + races + results を1行=1出走馬 で結合。"""
    where = f"WHERE r.race_date >= '{min_date}'" if min_date else ""
    df = pd.read_sql_query(f"""
        SELECT
            e.*,
            r.race_date, r.baba_code, r.race_no, r.distance_m, r.surface,
            r.direction, r.class_rank, r.track_cond_num, r.n_starters, r.grade,
            res.finish_pos, res.time_sec, res.last_3f,
            res.corner_1, res.corner_2, res.corner_3, res.corner_4,
            res.final_win_odds, res.final_popularity
        FROM entries e
        JOIN races r   USING (race_id)
        LEFT JOIN results res ON res.race_id = e.race_id
                             AND res.horse_no = e.horse_no
        {where}
        ORDER BY r.race_date, e.race_id, e.horse_no
    """, con, parse_dates=["race_date"])
    return df


def add_past_form(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    """
    馬ごとの過去走特徴。shift(1) で必ず1走ぶん過去にずらす。
    df は race_date 昇順に並んでいる前提。
    """
    df = df.sort_values(["horse_id", "race_date", "race_id"]).copy()
    g = df.groupby("horse_id", sort=False)

    # --- 前走情報 ---
    df["prev_date"] = g["race_date"].shift(1)
    df["days_rest"] = (df["race_date"] - df["prev_date"]).dt.days
    df["prev_finish"] = g["finish_pos"].shift(1)
    df["prev_corner4"] = g["corner_4"].shift(1)
    df["prev_class_rank"] = g["class_rank"].shift(1)
    df["prev_baba"] = g["baba_code"].shift(1)
    df["prev_dist"] = g["distance_m"].shift(1)
    df["prev_popularity"] = g["final_popularity"].shift(1)

    df["dist_change"] = df["distance_m"] - df["prev_dist"]
    df["class_move"] = df["prev_class_rank"] - df["class_rank"]  # +で昇級
    df["is_venue_change"] = (df["baba_code"] != df["prev_baba"]).astype(int)
    df["is_debut"] = df["prev_date"].isna().astype(int)

    # --- 移動平均系。shift(1) を先に噛ませてから rolling ---
    for w in windows:
        for src, name in [("finish_pos", "fin"), ("last_3f", "ag3"),
                          ("final_popularity", "pop")]:
            df[f"{name}_avg{w}"] = (
                g[src].shift(1)
                      .rolling(w, min_periods=1)
                      .mean()
                      .reset_index(level=0, drop=True)
            )
        # 着内率
        inm = (g["finish_pos"].shift(1) <= 3).astype(float)
        df[f"top3_rate{w}"] = (inm.rolling(w, min_periods=1).mean()
                                  .reset_index(level=0, drop=True))

    # --- 脚質推定: 過去の4角通過順を頭数で正規化した平均 ---
    pos_ratio = g["corner_4"].shift(1) / g["n_starters"].shift(1)
    df["running_style"] = (pos_ratio.rolling(5, min_periods=1).mean()
                                    .reset_index(level=0, drop=True))
    df["is_front_runner"] = (df["running_style"] < 0.25).astype(int)

    # --- 通算 ---
    df["career_starts"] = g.cumcount()
    df["career_wins"] = (g["finish_pos"].shift(1).eq(1).astype(float)
                          .groupby(df["horse_id"]).cumsum().fillna(0))
    df["career_win_rate"] = df["career_wins"] / df["career_starts"].replace(0, np.nan)

    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レース内の相対特徴 + 「荒れる」構造要因。
    これらは当該レースの出走表のみから作れる = リークしない。
    """
    gr = df.groupby("race_id", sort=False)

    # 相対化(絶対値より相対順位が効く)
    for c in ["fin_avg5", "top3_rate5", "career_win_rate", "weight_carried"]:
        if c in df:
            df[f"{c}_rank"] = gr[c].rank(pct=True)

    # --- 荒れ指数 ---
    # ① ハナ争い: 先行タイプが何頭いるか
    df["n_front_in_race"] = gr["is_front_runner"].transform("sum")
    df["front_congestion"] = df["n_front_in_race"] / df["n_starters"]

    # ② 先行馬の枠の偏り(外に固まるほど共倒れしやすい)
    outer = (df["horse_no"] - 1) / (df["n_starters"] - 1).clip(lower=1)
    df["front_outer_bias"] = (
        (outer * df["is_front_runner"]).groupby(df["race_id"]).transform("sum")
        / df["n_front_in_race"].replace(0, np.nan)
    )

    # ③ メンバーの実績のばらつき(小さい=どんぐり=荒れる)
    df["field_spread"] = gr["fin_avg5"].transform("std")

    # ④ 初出走・長期休養明けの比率(情報が薄い=荒れる)
    df["unknown_ratio"] = gr["is_debut"].transform("mean")
    df["layoff_ratio"] = gr.apply(
        lambda x: (x["days_rest"] > 90).mean(), include_groups=False
    ).reindex(df["race_id"]).values

    return df


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    市場の暗黙確率。オッズはレース前に確定するのでリークではない。
    ただしモデルの説明変数に入れるか外すかは目的で変わる:
      - EVを狙う   -> 入れない。市場と独立な自前推定を作り、後で突き合わせる
      - 的中率を狙う -> 入れる。ただしそれは市場の追認にしかならない
    今回の目的は前者なので、既定では別列として持つだけで学習には使わない。
    """
    inv = 1.0 / df["final_win_odds"].replace(0, np.nan)
    total = inv.groupby(df["race_id"]).transform("sum")
    df["mkt_prob"] = inv / total            # 控除率を戻した市場確率
    df["overround"] = total                 # 1.20〜1.35 程度が正常
    return df


FEATURE_COLS = [
    # 条件
    "distance_m", "class_rank", "track_cond_num", "n_starters",
    "horse_no", "bracket_no", "age", "weight_carried",
    "body_weight", "body_weight_dif",
    # 過去走
    "days_rest", "prev_finish", "prev_corner4", "prev_popularity",
    "dist_change", "class_move", "is_venue_change", "is_debut",
    "fin_avg3", "fin_avg5", "fin_avg10",
    "ag3_avg3", "ag3_avg5", "pop_avg5",
    "top3_rate3", "top3_rate5", "top3_rate10",
    "running_style", "is_front_runner",
    "career_starts", "career_win_rate",
    # 相対
    "fin_avg5_rank", "top3_rate5_rank", "career_win_rate_rank",
    "weight_carried_rank",
    # 荒れ指数
    "front_congestion", "front_outer_bias", "field_spread",
    "unknown_ratio", "layoff_ratio",
]


def build(con, min_date: str | None = None) -> pd.DataFrame:
    df = load_frame(con, min_date)
    df = add_past_form(df)
    df = add_context_features(df)
    df = add_market(df)
    df["y_win"] = (df["finish_pos"] == 1).astype(int)
    df["y_top3"] = (df["finish_pos"] <= 3).astype(int)
    return df


def assert_no_leak(df: pd.DataFrame, cols=FEATURE_COLS) -> None:
    """特徴量に結果由来の列が混入していないか機械的に検査する。
    人間のレビューは必ず失敗するので、必ずこれを CI に入れること。"""
    bad = set(cols) & LEAK_COLS
    if bad:
        raise AssertionError(f"LEAK: 結果由来の列が特徴量に入っている: {bad}")
    for c in cols:
        if c not in df.columns:
            raise AssertionError(f"missing feature column: {c}")
    print(f"leak check ok ({len(cols)} features)")


def time_split(df: pd.DataFrame, cutoff: str):
    """時系列分割。ランダム分割は絶対に使わないこと。
    同一レースが train/test に跨るだけでスコアが虚構になる。"""
    tr = df[df["race_date"] < cutoff]
    te = df[df["race_date"] >= cutoff]
    print(f"train {len(tr):,}行 / test {len(te):,}行  (cutoff={cutoff})")
    return tr, te
