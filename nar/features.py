"""
特徴量生成。

★このファイルで最も重要なのはリーク防止★
  「その馬の戦績」を使う以上、当該レース時点で知り得た情報しか使ってはいけない。
  ここを間違えると、バックテストで回収率150%が出て、実運用で70%に落ちる。

  対策:
    - 全ての集計に「当該行より過去」の厳格な条件を入れる (shift(1))
    - 走破タイム・上がり3F・通過順は当該レースのものは絶対に使わない
    - 馬体重・単勝オッズは当日だが「レース前に確定」なので使用可

★2026-08-15 の診断で判明した3つの問題を、このファイルで直している★

  (1) rolling がグループを跨いでいた
      旧: g[src].shift(1).rolling(w).mean()
          groupby.shift() はグループ情報を持たない素の Series を返すので、
          その後の rolling は馬の境界を無視して前の馬の値を混ぜていた。
          平均8.1走・窓10なら、ほぼ全馬の移動平均が汚染される。
      新: shift の後に必ず .groupby(horse_id) を挟む。

  (2) レース内で一定の特徴は、この模型では必ず消える
      model.ConditionalLogit は特徴をレース内 z化する。
      distance_m / n_starters / track_cond_num / class_rank / field_spread /
      unknown_ratio / layoff_ratio / front_congestion は
      レース内で全馬同じ値なので sd=0 → z=NaN → 0 になる。
      40特徴のうち16が「全NULL」か「レース内定数」で、実質24しか効いていなかった。
      → 特徴は「同じレースの馬の間で差がつくもの」でなければ意味がない。
        レース単位の量は class_move のように馬ごとの差分にして使う。

  (3) 死んでいた列を、取れるものだけ復活させた
      class_rank … race_name から復元 (raceclass.py)
      corner_1..4 … ★成績ページに存在しない★ ので復活不能。
                     脚質はタイムと上がり3Fの関係から間接的に取る。
      sex/age/body_weight … 成績ページに無い。出馬表の追加取得が要る。
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np

from . import raceclass as RC

# 当該レースの結果由来で、特徴量に混ぜてはいけない列
LEAK_COLS = {"finish_pos", "time_sec", "margin", "last_3f",
             "corner_1", "corner_2", "corner_3", "corner_4", "dnf_reason",
             # 下は当該レースの結果から作った中間量。lag 版だけを使う
             "spd", "ag3fig"}


def load_frame(con: sqlite3.Connection, min_date: str | None = None) -> pd.DataFrame:
    """entries + races + results を1行=1出走馬 で結合。"""
    where = f"WHERE r.race_date >= '{min_date}'" if min_date else ""
    df = pd.read_sql_query(f"""
        SELECT
            e.*,
            r.race_date, r.baba_code, r.race_no, r.distance_m, r.surface,
            r.direction, r.track_cond_num, r.n_starters, r.grade,
            r.race_name,
            res.finish_pos, res.time_sec, res.last_3f,
            res.final_win_odds, res.final_popularity
        FROM entries e
        JOIN races r   USING (race_id)
        LEFT JOIN results res ON res.race_id = e.race_id
                             AND res.horse_no = e.horse_no
        {where}
        ORDER BY r.race_date, e.race_id, e.horse_no
    """, con, parse_dates=["race_date"])
    return df.reset_index(drop=True)


# =====================================================================
# クラス復元
# =====================================================================
def add_class(df: pd.DataFrame) -> pd.DataFrame:
    """races.class_rank は全NULL。race_name から作り直す。"""
    uniq = df[["race_name", "baba_code"]].drop_duplicates()
    uniq["class_code"] = [
        RC.classify(n, b)[0] for n, b in
        zip(uniq["race_name"], uniq["baba_code"])
    ]
    uniq["class_rank"] = uniq["class_code"].map(RC.CLASS_RANK)
    uniq["is_age_race"] = uniq["class_code"].isin(RC.AGE_CODES).astype(int)
    df = df.merge(uniq, on=["race_name", "baba_code"], how="left")
    return df


# =====================================================================
# スピード指数
#   ★当該レースの spd は特徴量にしない。lag 版だけを使う★
#   したがって spd の計算に同日の他レースを使ってよい。
#   (spd はレースが終わった後に確定し、次走の材料になる量なので)
# =====================================================================
def add_speed_figure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["race_date", "race_id"]).copy()

    # --- コース基準タイム: (場, 馬場, 距離) ごとの累積平均。必ず shift(1) ---
    key = (df["baba_code"].astype(str) + "|" + df["surface"].astype(str)
           + "|" + df["distance_m"].astype(str))
    df["_ckey"] = key
    g = df.groupby("_ckey", sort=False)["time_sec"]
    base = g.transform(lambda s: s.shift(1).expanding().mean())
    sd = g.transform(lambda s: s.shift(1).expanding().std())
    sd = sd.where(sd > 0.01)

    dev = (df["time_sec"] - base) / sd            # 正 = 遅い

    # --- トラックバリアント: その日その場の全出走の中央値 ---
    #     馬場の速さ・天候をこれで吸収する
    variant = dev.groupby(
        [df["race_date"].values, df["baba_code"].values]).transform("median")
    df["spd"] = -(dev - variant)                  # 正 = 速い

    # --- 上がり3F も同じ手順で指数化 (末脚) ---
    g3 = df.groupby("_ckey", sort=False)["last_3f"]
    b3 = g3.transform(lambda s: s.shift(1).expanding().mean())
    s3 = g3.transform(lambda s: s.shift(1).expanding().std())
    s3 = s3.where(s3 > 0.01)
    dev3 = (df["last_3f"] - b3) / s3
    var3 = dev3.groupby(
        [df["race_date"].values, df["baba_code"].values]).transform("median")
    df["ag3fig"] = -(dev3 - var3)

    # --- 前半の速さ = 全体の速さ - 末脚の速さ。脚質の代理指標 ---
    #     corner_4 が取れない以上、逃げ・追込はここから推定するしかない
    df["pacefig"] = df["spd"] - df["ag3fig"]

    return df.drop(columns="_ckey")


# =====================================================================
# 馬ごとの過去走
# =====================================================================
def _roll(s: pd.Series, by: pd.Series, w: int, how: str) -> pd.Series:
    """shift済みの系列を、必ずグループを跨がずに rolling する。

    ★旧実装の事故はここ★
      groupby.shift() の戻りはグループ情報を持たない。
      そのまま rolling すると前の馬の値が混ざる。
    """
    r = s.groupby(by).rolling(w, min_periods=1)
    out = getattr(r, how)()
    return out.reset_index(level=0, drop=True)


def add_past_form(df: pd.DataFrame, windows=(3, 5, 10)) -> pd.DataFrame:
    df = df.sort_values(["horse_id", "race_date", "race_id"]).copy()
    hid = df["horse_id"]
    g = df.groupby("horse_id", sort=False)

    # --- 前走 ---
    df["prev_date"] = g["race_date"].shift(1)
    df["days_rest"] = (df["race_date"] - df["prev_date"]).dt.days
    df["prev_finish"] = g["finish_pos"].shift(1)
    df["prev_popularity"] = g["final_popularity"].shift(1)
    df["prev_class_rank"] = g["class_rank"].shift(1)
    df["prev_baba"] = g["baba_code"].shift(1)
    df["prev_dist"] = g["distance_m"].shift(1)

    df["dist_change"] = df["distance_m"] - df["prev_dist"]
    df["class_move"] = df["prev_class_rank"] - df["class_rank"]  # +で昇級
    df["is_venue_change"] = (df["baba_code"] != df["prev_baba"]).astype(int)
    df["is_debut"] = df["prev_date"].isna().astype(int)

    # --- 移動平均。shift してから必ず再グループ化する ---
    for w in windows:
        for src, name in [("finish_pos", "fin"), ("last_3f", "ag3"),
                          ("final_popularity", "pop")]:
            df[f"{name}_avg{w}"] = _roll(g[src].shift(1), hid, w, "mean")
        inm = (g["finish_pos"].shift(1) <= 3).astype(float)
        df[f"top3_rate{w}"] = _roll(inm, hid, w, "mean")

    # --- スピード指数の過去走版 (当該レースの spd は絶対に使わない) ---
    sp = g["spd"].shift(1)
    df["spd_last"] = sp
    df["spd_avg3"] = _roll(sp, hid, 3, "mean")
    df["spd_avg5"] = _roll(sp, hid, 5, "mean")
    df["spd_best5"] = _roll(sp, hid, 5, "max")
    df["spd_std5"] = _roll(sp, hid, 5, "std")
    df["spd_trend"] = df["spd_last"] - df["spd_avg5"]

    ag = g["ag3fig"].shift(1)
    df["ag3fig_avg3"] = _roll(ag, hid, 3, "mean")
    pc = g["pacefig"].shift(1)
    df["pacefig_avg5"] = _roll(pc, hid, 5, "mean")

    # --- 距離・場の経験 ---
    same_d = (g["distance_m"].shift(1) == df["distance_m"]).astype(float)
    df["dist_experience"] = _roll(same_d, hid, 10, "mean")

    # --- 通算 ---
    df["career_starts"] = g.cumcount()
    win = (df["finish_pos"] == 1).astype(float)
    df["career_wins"] = win.groupby(hid).cumsum() - win     # 当該走を除く
    df["career_win_rate"] = (
        df["career_wins"] / df["career_starts"].replace(0, np.nan))

    return df


# =====================================================================
# 騎手・調教師の成績
#   jockey_name / trainer_name は 99.8% 埋まっている。追加取得は不要。
#   IDが無くても名前で名寄せできる。
# =====================================================================
def add_person_form(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["race_date", "race_id", "horse_no"]).copy()
    win = (d["finish_pos"] == 1).astype(float)
    top3 = (d["finish_pos"] <= 3).astype(float)

    for col, pre in [("jockey_name", "jky"), ("trainer_name", "trn")]:
        k = d[col].fillna("__na__")
        n = d.groupby(k, sort=False).cumcount()          # 当該走より前の騎乗数
        cw = win.groupby(k).cumsum() - win
        ct = top3.groupby(k).cumsum() - top3
        d[f"{pre}_starts"] = n
        d[f"{pre}_win_rate"] = cw / n.replace(0, np.nan)
        d[f"{pre}_top3_rate"] = ct / n.replace(0, np.nan)

    # 騎手の直近200騎乗の勝率 (通算だとベテランが有利に出すぎる)
    k = d["jockey_name"].fillna("__na__")
    d["jky_recent_win"] = _roll(win.groupby(k).shift(1), k, 200, "mean")

    # この騎手はこの馬に乗ったことがあるか (乗り慣れ)
    pair = d["horse_id"].astype(str) + "|" + k
    d["jky_horse_rides"] = d.groupby(pair, sort=False).cumcount()

    new = [c for c in d.columns if c not in df.columns]
    return df.join(d[new])


# =====================================================================
# レース内の相対化
#   ★この模型はレース内 z化するので、レース内で一定の量は必ず消える★
#   ここで作るのは「同じレースの他馬と比べてどうか」だけ。
# =====================================================================
REL_COLS = ["fin_avg5", "top3_rate5", "career_win_rate", "weight_carried",
            "spd_best5", "spd_avg5", "ag3fig_avg3", "pacefig_avg5",
            "jky_win_rate", "jky_recent_win", "trn_win_rate", "days_rest"]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    gr = df.groupby("race_id", sort=False)
    for c in REL_COLS:
        if c in df:
            df[f"{c}_rank"] = gr[c].rank(pct=True)

    # 相手関係: 自分のスピードとレース最高スピードの差
    if "spd_best5" in df:
        df["spd_gap_to_top"] = df["spd_best5"] - gr["spd_best5"].transform("max")
    return df


def add_market(df: pd.DataFrame) -> pd.DataFrame:
    """市場の暗黙確率。学習には使わず、比較用に持つだけ。"""
    inv = 1.0 / df["final_win_odds"].replace(0, np.nan)
    total = inv.groupby(df["race_id"]).transform("sum")
    df["mkt_prob"] = inv / total
    df["overround"] = total
    return df


# =====================================================================
# ★特徴量は「レース内で差がつくもの」に限る★
#   レース内定数 (distance_m, n_starters, class_rank, track_cond_num,
#   field_spread, unknown_ratio, layoff_ratio) は z化で消えるので入れない。
# =====================================================================
FEATURE_COLS = [
    # 枠・斤量
    "horse_no", "bracket_no", "weight_carried",
    # 前走との関係 (馬ごとに違う = 生き残る)
    "days_rest", "prev_finish", "prev_popularity",
    "dist_change", "class_move", "is_venue_change", "is_debut",
    "dist_experience",
    # 着順・人気の移動平均
    "fin_avg3", "fin_avg5", "fin_avg10",
    "ag3_avg3", "ag3_avg5", "pop_avg5",
    "top3_rate3", "top3_rate5", "top3_rate10",
    "career_starts", "career_win_rate",
    # スピード指数 (新規。タイム由来の特徴はこれまでゼロだった)
    "spd_last", "spd_avg3", "spd_avg5", "spd_best5", "spd_std5", "spd_trend",
    "ag3fig_avg3", "pacefig_avg5",
    # 騎手・調教師 (新規)
    "jky_starts", "jky_win_rate", "jky_top3_rate", "jky_recent_win",
    "jky_horse_rides", "trn_win_rate", "trn_top3_rate",
    # レース内相対
    "fin_avg5_rank", "top3_rate5_rank", "career_win_rate_rank",
    "weight_carried_rank", "spd_best5_rank", "spd_avg5_rank",
    "ag3fig_avg3_rank", "pacefig_avg5_rank",
    "jky_win_rate_rank", "jky_recent_win_rank", "trn_win_rate_rank",
    "days_rest_rank", "spd_gap_to_top",
]

# レース内で一定なので、この模型では必ず0になる列。
# 入れても害は無いが効果も無い。木モデルに替えるときは復活させてよい。
RACE_CONSTANT_COLS = [
    "distance_m", "n_starters", "track_cond_num", "class_rank",
    "is_age_race", "field_spread", "unknown_ratio", "layoff_ratio",
]


def build(con, min_date: str | None = None) -> pd.DataFrame:
    df = load_frame(con, min_date)
    df = add_class(df)
    df = add_speed_figure(df)
    df = add_past_form(df)
    df = add_person_form(df)
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


def assert_varies_within_race(df: pd.DataFrame, cols=FEATURE_COLS,
                              sample: int = 2000) -> None:
    """レース内で一定の特徴を検出する。

    model.ConditionalLogit はレース内 z化するので、
    レース内定数は sd=0 → 全行0 になり、学習に一切寄与しない。
    「40特徴あるつもりで実は24」という事故を防ぐ。
    """
    ids = df["race_id"].drop_duplicates().head(sample)
    sub = df[df["race_id"].isin(ids)]
    g = sub.groupby("race_id")
    dead = []
    for c in cols:
        if c not in sub:
            continue
        sd = g[c].std()
        if not (sd.fillna(0) > 1e-12).any():
            dead.append(c)
    if dead:
        print(f"  !! レース内で一定 = z化で消える特徴 {len(dead)}件: {dead}")
    else:
        print(f"  レース内変動チェック ok ({len(cols)}特徴すべて差がつく)")


def coverage_report(df: pd.DataFrame, cols=FEATURE_COLS) -> None:
    """特徴ごとの非NULL率。全NULLの列が紛れていないか確認する。"""
    print("\n  特徴量の充足率 (低い順15件):")
    r = [(c, df[c].notna().mean() * 100) for c in cols if c in df]
    for c, p in sorted(r, key=lambda x: x[1])[:15]:
        mark = "  ★全NULL" if p == 0 else ""
        print(f"    {c:<22} {p:5.1f}%{mark}")


def time_split(df: pd.DataFrame, cutoff: str):
    """時系列分割。ランダム分割は絶対に使わないこと。"""
    tr = df[df["race_date"] < cutoff]
    te = df[df["race_date"] >= cutoff]
    print(f"train {len(tr):,}行 / test {len(te):,}行  (cutoff={cutoff})")
    return tr, te
