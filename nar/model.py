# ===== nar/model.py     (条件付きロジット + 温度較正 + EV) =====
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
from scipy.optimize import minimize_scalar
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
        self.temperature: float = 1.0
        self.calib_cut: str | None = None

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

    def fit(self, df: pd.DataFrame, cols: list[str],
            calib_frac: float = 0.15) -> "ConditionalLogit":
        """
        calib_frac > 0 なら、学習データの★時系列で後ろ側★を温度較正に使う。
        ランダムに切ると未来を見てしまうので必ず日付順の末尾を取る。
        """
        self.cols = cols
        fit_df, cal_df = df, df.iloc[:0]
        if calib_frac > 0:
            dates = df["race_date"].drop_duplicates().sort_values()
            cut = dates.iloc[int(len(dates) * (1 - calib_frac))]
            a = df[df["race_date"] < cut]
            b = df[df["race_date"] >= cut]
            if len(a) >= 5000 and len(b) >= 2000:
                fit_df, cal_df = a, b
                self.calib_cut = str(cut)[:10]

        X = self._within_race(fit_df, cols)
        X = self.scaler.fit_transform(X.values)
        y = (fit_df["finish_pos"] == 1).astype(int).values
        self.clf.fit(X, y)

        if len(cal_df):
            self.temperature = self._fit_temperature(cal_df)
        return self

    # -----------------------------------------------------------------
    # 温度較正
    #   softmax は尖りすぎることがある。実測では最上位ビンで
    #   予測47.2% に対し実現24.9% だった。EV = p x odds なので
    #   ここがズレていると EV の数字は全部嘘になる。
    #   スカラー1個 T を入れて softmax(s/T) の対数尤度を最大化する。
    # -----------------------------------------------------------------
    def _fit_temperature(self, df: pd.DataFrame) -> float:
        s = self.score(df)
        y = (df["finish_pos"] == 1).astype(int).values
        rid = df["race_id"]

        def nll(logT: float) -> float:
            p = softmax_by_race(df, s / np.exp(logT))
            p = np.clip(p, 1e-12, 1.0)
            return -np.log(p[y == 1]).sum() / max(y.sum(), 1)

        r = minimize_scalar(nll, bounds=(np.log(0.3), np.log(6.0)),
                            method="bounded")
        return float(np.exp(r.x))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._within_race(df, self.cols)
        X = self.scaler.transform(X.values)
        return self.clf.decision_function(X)

    def predict_proba(self, df: pd.DataFrame, raw: bool = False) -> np.ndarray:
        """レース内ソフトマックス。レースごとに合計1になる。
        raw=True で温度較正を外した生の確率を返す(較正の効果測定用)。"""
        t = 1.0 if raw else self.temperature
        return softmax_by_race(df, self.score(df) / t)

    def coef_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.cols,
                              "coef": self.clf.coef_[0]})
                .assign(abs_coef=lambda d: d["coef"].abs())
                .sort_values("abs_coef", ascending=False)
                .drop(columns="abs_coef"))


# =====================================================================
# 厳密な条件付きロジット (Plackett-Luce の1着部分)
#
# ★上の ConditionalLogit との違い★
#   上は「レース内z化 → 二値ロジスティック → 予測時にsoftmax」という近似。
#   これには2つの副作用がある:
#
#   (1) 二値ロジスティックが最適化しているのは
#       「この馬は勝つか」であって「レースの中で誰が勝つか」ではない。
#       係数の尺度が softmax 用に較正されないので、確率が尖りすぎる。
#
#   (2) レース内 std で割ると、レースごとの「差の大きさ」が消える。
#       抜けた1頭がいるレースも横一線のレースも同じ尺度に潰れるので、
#       自信を持つべき場面と持つべきでない場面の区別がつかなくなる。
#       2026-08-15 の実測でキャリブレーション最大乖離 0.20
#       (予測47% vs 実現25%) が残ったのはこれが原因の可能性が高い。
#
#   ここでは尤度を直接最大化する:
#       NLL = -Σ_race [ x_win·β - logΣ_j exp(x_j·β) ] + λ|β|²
#   レース内の差だけが効くので、レース単位の定数は自動的に落ちる。
#   標準化は全体で行い、レース内では中心化のみ(数値安定のため)。
# =====================================================================
class ConditionalLogitExact:

    def __init__(self, C: float = 1.0, center: bool = True):
        self.C = C
        self.center = center
        self.scaler = StandardScaler()
        self.cols: list[str] = []
        self.beta: np.ndarray | None = None
        self.temperature: float = 1.0
        self.calib_cut: str | None = None
        self.n_iter_: int = 0

    def _design(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = df[self.cols].astype(float).fillna(0.0).values
        X = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        if self.center:
            # レース内で中心化。softmax では定数が消えるので情報は失わない。
            # ★std で割らないのが上との決定的な違い★
            t = pd.DataFrame(X, index=df.index)
            X = (t - t.groupby(df["race_id"].values).transform("mean")).values
        return np.nan_to_num(X)

    @staticmethod
    def _group_index(rid: pd.Series) -> np.ndarray:
        codes, _ = pd.factorize(rid.values)
        return codes

    def fit(self, df: pd.DataFrame, cols: list[str],
            calib_frac: float = 0.15) -> "ConditionalLogitExact":
        self.cols = cols
        fit_df, cal_df = df, df.iloc[:0]
        if calib_frac > 0:
            dates = df["race_date"].drop_duplicates().sort_values()
            cut = dates.iloc[int(len(dates) * (1 - calib_frac))]
            a, b = df[df["race_date"] < cut], df[df["race_date"] >= cut]
            if len(a) >= 5000 and len(b) >= 2000:
                fit_df, cal_df = a, b
                self.calib_cut = str(cut)[:10]

        X = self._design(fit_df, fit=True)
        g = self._group_index(fit_df["race_id"])
        y = (fit_df["finish_pos"] == 1).astype(int).values.astype(bool)
        ng = g.max() + 1
        lam = 1.0 / (2.0 * self.C * max(ng, 1))

        Xw = np.zeros((ng, X.shape[1]))
        np.add.at(Xw, g[y], X[y])          # レースごとの勝ち馬の特徴

        def obj(beta):
            s = X @ beta
            m = np.full(ng, -np.inf)
            np.maximum.at(m, g, s)
            e = np.exp(s - m[g])
            Z = np.bincount(g, weights=e, minlength=ng)
            lse = m + np.log(Z)
            nll = float(-(Xw @ beta).sum() + lse.sum()) / ng \
                + lam * float(beta @ beta)
            p = e / Z[g]
            Xp = np.zeros_like(Xw)
            np.add.at(Xp, g, X * p[:, None])
            grad = -(Xw.sum(0) - Xp.sum(0)) / ng + 2 * lam * beta
            return nll, grad

        from scipy.optimize import minimize
        r = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                     options={"maxiter": 500})
        self.beta = r.x
        self.n_iter_ = int(r.nit)

        if len(cal_df):
            self.temperature = self._fit_temperature(cal_df)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self._design(df) @ self.beta

    def _fit_temperature(self, df: pd.DataFrame) -> float:
        s = self.score(df)
        y = (df["finish_pos"] == 1).astype(int).values

        def nll(logT):
            p = np.clip(softmax_by_race(df, s / np.exp(logT)), 1e-12, 1.0)
            return -np.log(p[y == 1]).sum() / max(y.sum(), 1)

        r = minimize_scalar(nll, bounds=(np.log(0.3), np.log(6.0)),
                            method="bounded")
        return float(np.exp(r.x))

    def predict_proba(self, df: pd.DataFrame, raw: bool = False) -> np.ndarray:
        t = 1.0 if raw else self.temperature
        return softmax_by_race(df, self.score(df) / t)

    def coef_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.cols, "coef": self.beta})
                .assign(a=lambda d: d["coef"].abs())
                .sort_values("a", ascending=False).drop(columns="a"))


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


# =====================================================================
# 払戻ベースのバックテスト (単勝以外も検証できる)
#
# ★これが使える理由★
#   オッズは3,412レース分しか無いが、payouts は 198,072行あり
#   全16,467レースを覆っている。
#   買い目を決めてから「当たったらいくら戻るか」を見るだけなら、
#   事前オッズは要らない。的中した組の払戻さえ分かればよい。
#   → 複勝・ワイド・三連複などを全レースで検証できる。
#
#   制約: 事前オッズが無いので EV 閾値は掛けられない。
#         「常に買う」戦略の回収率しか測れない。
#         それでも「単勝以外に妙味があるか」の一次判定には十分。
# =====================================================================
UNORDERED = {"place", "quinella", "wide", "trio"}

BET_PICK = {                      # (券種, 上位何頭を使うか)
    "win":      1,
    "place":    1,
    "quinella": 2,
    "exacta":   2,
    "wide":     2,
    "trio":     3,
    "trifecta": 3,
}


def load_payout_map(con) -> dict:
    """(race_id, bet_type, 正規化した組) -> 払戻円"""
    out = {}
    for rid, bt, comb, yen in con.execute(
            "SELECT race_id, bet_type, combination, payout_yen FROM payouts"):
        key = comb
        if bt in UNORDERED:
            key = "-".join(sorted(comb.split("-"), key=int))
        out[(rid, bt, key)] = yen
    return out


def backtest_payouts(df: pd.DataFrame, p: np.ndarray, pmap: dict,
                     bets=("win", "place", "quinella", "exacta",
                           "wide", "trio", "trifecta"),
                     stake: float = 100.0, label: str = "model",
                     seed: int = 0) -> pd.DataFrame:
    """各レースで確率上位n頭を1点だけ買う。回収率を券種ごとに出す。"""
    d = pd.DataFrame({
        "race_id": df["race_id"].values,
        "horse_no": df["horse_no"].values,
        "p": p,
    }).sort_values(["race_id", "p"], ascending=[True, False])
    picks = d.groupby("race_id")["horse_no"].apply(list)

    rng = np.random.default_rng(seed)
    rows = []
    for bt in bets:
        k = BET_PICK[bt]
        rets, n = [], 0
        for rid, hs in picks.items():
            if len(hs) < k:
                continue
            sel = [str(int(x)) for x in hs[:k]]
            key = "-".join(sorted(sel, key=int)) if bt in UNORDERED \
                else "-".join(sel)
            n += 1
            rets.append(float(pmap.get((rid, bt, key), 0)))
        if not n:
            continue
        r = np.array(rets)
        bet_total = n * stake
        boots = [r[rng.integers(0, n, n)].sum() / bet_total
                 for _ in range(300)] if n >= 30 else []
        rows.append({
            "戦略": label, "券種": bt, "点数": n,
            "的中": int((r > 0).sum()), "的中率%": (r > 0).mean() * 100,
            "回収率%": r.sum() / bet_total * 100,
            "95%下限": np.percentile(boots, 2.5) * 100 if boots else np.nan,
            "95%上限": np.percentile(boots, 97.5) * 100 if boots else np.nan,
        })
    return pd.DataFrame(rows)



# =====================================================================
# 市場そのものの効率性マップ
#
# ★モデルを作る前に確かめるべきだったこと★
#   「どこで市場が間違えているか」を先に測る。
#   全区分で回収率が控除率なりに沈んでいるなら、
#   その券種でEV100%超を狙う前提そのものが崩れる。
#
#   人気順位は100%埋まっているので、オッズが無い期間も含めた
#   全16,467レースで測れる。払戻テーブルがあれば足りる。
# =====================================================================
def market_efficiency(df: pd.DataFrame, pmap: dict,
                      bets=("win", "place"), seed: int = 0) -> pd.DataFrame:
    """人気順位ごとの回収率。単純に『N番人気を毎回買う』戦略の成績。"""
    rng = np.random.default_rng(seed)
    rows = []
    for bt in bets:
        for pop, g in df.groupby("final_popularity"):
            if pop is None or pop != pop or pop > 14 or len(g) < 100:
                continue
            ret = np.array([
                float(pmap.get((r, bt, str(int(h))), 0))
                for r, h in zip(g["race_id"], g["horse_no"])
            ])
            n = len(ret)
            boots = [ret[rng.integers(0, n, n)].mean()
                     for _ in range(300)] if n >= 100 else []
            rows.append({
                "券種": bt, "人気": int(pop), "点数": n,
                "的中率%": (ret > 0).mean() * 100,
                "回収率%": ret.mean(),
                "95%下限": np.percentile(boots, 2.5) if boots else np.nan,
                "95%上限": np.percentile(boots, 97.5) if boots else np.nan,
            })
    return pd.DataFrame(rows)


def market_efficiency_by(df: pd.DataFrame, pmap: dict, key: str,
                         bet: str = "win", top_pop: int = 1,
                         seed: int = 0) -> pd.DataFrame:
    """区分ごとに『top_pop番人気を買う』回収率。区分の偏りを探す。"""
    rng = np.random.default_rng(seed)
    sub = df[df["final_popularity"] == top_pop]
    rows = []
    for k, g in sub.groupby(key):
        if len(g) < 100:
            continue
        ret = np.array([
            float(pmap.get((r, bet, str(int(h))), 0))
            for r, h in zip(g["race_id"], g["horse_no"])
        ])
        n = len(ret)
        boots = [ret[rng.integers(0, n, n)].mean() for _ in range(200)]
        rows.append({
            key: k, "点数": n, "的中率%": (ret > 0).mean() * 100,
            "回収率%": ret.mean(),
            "95%下限": np.percentile(boots, 2.5),
            "95%上限": np.percentile(boots, 97.5),
        })
    return pd.DataFrame(rows).sort_values("回収率%", ascending=False)


def time_split(df: pd.DataFrame, cutoff: str):
    tr = df[df["race_date"] < cutoff].copy()
    te = df[df["race_date"] >= cutoff].copy()
    return tr, te
