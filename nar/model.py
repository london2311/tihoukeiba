"""
着順確率モデル — 条件付きロジット (Plackett-Luce の1着部分)。

設計:
  各馬 i に強さスコア s_i を出し、レース内でソフトマックスして勝率にする。
      P(i が1着) = exp(s_i) / Σ_j exp(s_j)
  レース単位で確率が1に合うので EV 計算にそのまま使える。
  木モデルより素直で、地方の薄いサンプルでも壊れにくい。

学習と検証の分離 (これを間違えると全部無意味になる):
  学習   … 全16万出走。オッズは不要
  検証   … オッズ完備レースのみ。実オッズだけを使い推定値は混ぜない
  分割   … 必ず時系列。ランダム分割は禁止

合否判定:
  最終的に見るのは「市場(人気順)より賢いか」の一点。
  控除率18.8%を超える精度差が無ければ EV100%超は原理的に成立しない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss


def softmax_by_race(df: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    """レース内ソフトマックス。ベクトル化してあるので大きいデータでも速い。"""
    t = pd.DataFrame({"r": df["race_id"].values, "s": s})
    m = t.groupby("r")["s"].transform("max")
    e = np.exp(t["s"] - m)
    return (e / e.groupby(t["r"]).transform("sum")).values


# =====================================================================
# 学習
# =====================================================================
class ConditionalLogit:
    """
    レース内で相対化した特徴を使うロジスティック回帰。

    条件付きロジットの厳密推定は重いので、
      ① 特徴をレース内で標準化(z化)する
      ② 1着かどうかの二値ロジスティックを解く
      ③ 予測時にレース内ソフトマックスで正規化する
    という近似を使う。実務上ほぼ同等で、はるかに速く安定する。
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", class_weight=None)
        self.cols: list[str] = []

    @staticmethod
    def _within_race(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        レース内 z化。★これが条件付きロジットの肝★
        絶対値ではなく「同じレースの他馬と比べてどうか」だけを見る。
        レース間の水準差(クラス・時期・場)が自動的に落ちる。
        """
        g = df.groupby("race_id")[cols]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        z = (df[cols] - mu) / sd
        return z.fillna(0.0)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "ConditionalLogit":
        self.cols = cols
        X = self._within_race(df, cols)
        X = self.scaler.fit_transform(X.values)
        y = (df["finish_pos"] == 1).astype(int).values
        self.clf.fit(X, y)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._within_race(df, self.cols)
        X = self.scaler.transform(X.values)
        return self.clf.decision_function(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """レース内ソフトマックス。レースごとに合計1になる。"""
        return softmax_by_race(df, self.score(df))

    def coef_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.cols,
                              "coef": self.clf.coef_[0]})
                .assign(abs_coef=lambda d: d["coef"].abs())
                .sort_values("abs_coef", ascending=False)
                .drop(columns="abs_coef"))


# =====================================================================
# 市場ベースライン
# =====================================================================
def market_prob_from_popularity(df: pd.DataFrame) -> np.ndarray:
    """
    人気順位だけから作る素朴な市場確率。
    「人気順位 -> 勝率」の実測平均を頭数別に当てはめる。
    モデルはこれを上回れなければ意味がない。
    """
    t = df[["race_id", "n_starters", "final_popularity"]].copy()
    t["w"] = 1.0 / t["final_popularity"].clip(lower=1)
    tot = t.groupby("race_id")["w"].transform("sum")
    return (t["w"] / tot).values


def market_prob_from_odds(df: pd.DataFrame) -> np.ndarray:
    """実オッズから控除率を戻した市場確率。オッズ完備レースでのみ使える。"""
    inv = 1.0 / df["final_win_odds"].replace(0, np.nan)
    tot = inv.groupby(df["race_id"]).transform("sum")
    return (inv / tot).values


# =====================================================================
# 評価
# =====================================================================
def evaluate(name: str, y: np.ndarray, p: np.ndarray,
             race_id: pd.Series) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    d = {
        "name": name,
        "n": len(y),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    }
    # レース単位の的中率: 最高確率の馬が実際に1着か
    t = pd.DataFrame({"r": race_id.values, "p": p, "y": y})
    top = t.loc[t.groupby("r")["p"].idxmax()]
    d["top1_hit"] = top["y"].mean()
    return d


def calibration_table(y: np.ndarray, p: np.ndarray, bins=10) -> pd.DataFrame:
    """
    予測確率とその実現率の対応。★EVの前提はここが合っていること★
    「15%と言った馬が本当に15%勝つ」が崩れていたら、EVの数字は全部嘘になる。
    """
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "bin": q})
    g = t.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    g["diff"] = g["actual"] - g["pred"]
    return g.reset_index(drop=True)


# =====================================================================
# EV バックテスト (単勝・実オッズのみ)
# =====================================================================
def backtest_win(df: pd.DataFrame, p: np.ndarray,
                 ev_thresholds=(1.00, 1.05, 1.10, 1.20, 1.50),
                 stake: float = 100.0) -> pd.DataFrame:
    """
    EV = p * odds。閾値を超えた買い目だけ単勝を1点買いする。

    ★実オッズがある行だけを対象にすること★
    推定オッズを混ぜた瞬間にバックテストは信用できなくなる。
    """
    d = df.copy()
    d["p"] = p
    d = d[d["final_win_odds"].notna() & (d["final_win_odds"] > 0)]
    d["ev"] = d["p"] * d["final_win_odds"]
    d["hit"] = (d["finish_pos"] == 1).astype(int)
    d["ret"] = d["hit"] * d["final_win_odds"] * stake

    rows = []
    for th in ev_thresholds:
        s = d[d["ev"] >= th]
        if len(s) == 0:
            rows.append({"EV閾値": th, "点数": 0})
            continue
        bet = len(s) * stake
        ret = s["ret"].sum()
        # ブートストラップで回収率の95%区間を出す(点推定だけ見ると事故る)
        rng = np.random.default_rng(0)
        boots = [
            s["ret"].values[rng.integers(0, len(s), len(s))].sum() / bet
            for _ in range(400)
        ] if len(s) >= 30 else []
        rows.append({
            "EV閾値": th,
            "点数": len(s),
            "的中": int(s["hit"].sum()),
            "的中率%": s["hit"].mean() * 100,
            "回収率%": ret / bet * 100,
            "95%下限": np.percentile(boots, 2.5) * 100 if boots else np.nan,
            "95%上限": np.percentile(boots, 97.5) * 100 if boots else np.nan,
            "平均オッズ": s["final_win_odds"].mean(),
        })
    return pd.DataFrame(rows)


def time_split(df: pd.DataFrame, cutoff: str):
    tr = df[df["race_date"] < cutoff].copy()
    te = df[df["race_date"] >= cutoff].copy()
    return tr, te
