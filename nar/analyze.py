"""
収集済みDBの現状把握と、較正の検証。

    python -m nar.analyze --report      # データ概況 + 欠損診断
    python -m nar.analyze --calibrate   # 人気→オッズ較正 + 検証

--report は以下を機械的に洗い出す:
  A. 全テーブルの列ごと NULL 率 (どの特徴量が死んでいるか)
  B. race_name の実物 (場ごとに層化サンプル)
  C. race_name からのクラス抽出テスト (網羅率 + 未マッチ上位)
  D. パーサの取りこぼし検出 (斤量/着差/所属)
  E. corner_4 が全NULLなら、成績ページを1本だけ取得して
     コーナー通過順がHTML内に存在するか確認する
     ※ SQL では絶対に分からないので、ここでネットワークを1回だけ使う
"""
from __future__ import annotations

import argparse
import re
import unicodedata
import numpy as np
import pandas as pd

from .core import connect, BABA
from . import oddscal


# =====================================================================
# クラス判定 (race_name から)
#   NFKC 正規化で Ｂ２ -> B2 になる。全角half角の分岐を書かなくてよい。
# =====================================================================
CLASS_AB_RE = re.compile(r"(?<![A-Za-z])([ABCD])\s?([1-3])(?![0-9])")

GRADE_WORDS = [
    ("JpnI",   re.compile(r"JpnI(?!I)")),
    ("JpnII",  re.compile(r"JpnII(?!I)")),
    ("JpnIII", re.compile(r"JpnIII")),
    ("GI",     re.compile(r"(?<![A-Za-z])GI(?!I)")),
    ("重賞",   re.compile(r"重賞")),
]


def classify_race_name(name: str | None) -> tuple[str | None, str]:
    """race_name -> (class_code, どのルールで当たったか)

    返り値の class_code は core.CLASS_RANK のキーに合わせる。
    当たらなければ (None, 'unmatched')。
    """
    if not name:
        return None, "empty"
    s = unicodedata.normalize("NFKC", name)

    for label, rx in GRADE_WORDS:
        if rx.search(s):
            return label if label.startswith("Jpn") else "重賞", "grade"

    m = CLASS_AB_RE.search(s)
    if m:
        return f"{m.group(1)}{m.group(2)}", "AB123"

    # クラス記号なしの区分。CLASS_RANK には無いので別枠で数える
    if re.search(r"オープン|ＯＰ|OP(?![A-Za-z])", s):
        return "OPEN", "open"
    if "未勝利" in s:
        return "未勝利", "word"
    if "新馬" in s or "デビュー" in s:
        return "新馬", "word"
    if re.search(r"(?<![0-9])2歳", s):
        return "2歳", "age"
    if re.search(r"(?<![0-9])3歳", s):
        return "3歳", "age"
    return None, "unmatched"


# =====================================================================
# A. 列ごとの NULL 率
# =====================================================================
def null_report(con) -> None:
    print("\n" + "=" * 64)
    print("■ A. 列ごとの充足率  (100%未満のみ表示)")
    for table in ("races", "entries", "results"):
        total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if not total:
            print(f"\n  [{table}] 0行")
            continue
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        rows = []
        for c in cols:
            n = con.execute(
                f'SELECT COUNT("{c}") FROM {table}').fetchone()[0]
            if n < total:
                rows.append((c, n, n / total * 100))
        print(f"\n  [{table}] {total:,}行")
        if not rows:
            print("    全列100%")
            continue
        for c, n, pct in sorted(rows, key=lambda x: x[2]):
            mark = "  ★全NULL" if n == 0 else ""
            print(f"    {c:<18} {n:>9,}  {pct:5.1f}%{mark}")


# =====================================================================
# B/C. race_name の実物とクラス抽出テスト
# =====================================================================
def race_name_report(con) -> None:
    df = pd.read_sql_query(
        "SELECT race_id, race_date, baba_code, baba_name, race_no, race_name,"
        "       n_starters, distance_m FROM races", con)
    if df.empty:
        print("\n  races が空")
        return

    print("\n" + "=" * 64)
    print("■ B. race_name の実物 (場ごとに2件ずつ)")
    n_null = df["race_name"].isna().sum()
    n_blank = (df["race_name"].fillna("").str.strip() == "").sum()
    print(f"  NULL {n_null:,} / 空文字 {n_blank:,} / 全{len(df):,}件")

    for baba, g in df.groupby("baba_code"):
        name = BABA.get(int(baba), str(baba))
        for _, r in g.head(2).iterrows():
            print(f"    {name:<4}({int(baba):>2}) {r['race_date']} "
                  f"{int(r['race_no']):>2}R  {r['race_name']!r}")

    # 抽出ロジックが1行ずれていないかの傍証:
    # レース名に距離や「発走」「競馬場」が混じっていたら別行を拾っている
    susp = df[df["race_name"].fillna("").str.contains(
        r"競馬場|発走|ダ\(|芝\(|\d{3,4}m", regex=True)]
    if len(susp):
        print(f"\n  !! レース名に条件文字列が混入 {len(susp):,}件 "
              f"({len(susp)/len(df)*100:.1f}%) — 抽出行がずれている疑い")
        for v in susp["race_name"].head(5):
            print(f"     {v!r}")

    # ---------------- C ----------------
    print("\n" + "=" * 64)
    print("■ C. race_name からのクラス抽出テスト")
    res = df["race_name"].apply(classify_race_name)
    df["class_code"] = [x[0] for x in res]
    df["rule"] = [x[1] for x in res]

    hit = df["class_code"].notna().sum()
    ab = (df["rule"] == "AB123").sum()
    print(f"  何らかのクラスが取れた   {hit:>7,} / {len(df):,}  "
          f"({hit/len(df)*100:5.1f}%)")
    print(f"  うち A1..D2 の記号      {ab:>7,}          "
          f"({ab/len(df)*100:5.1f}%)   ← features で直接使えるのはここ")

    print("\n  ルール別:")
    for rule, n in df["rule"].value_counts().items():
        print(f"    {rule:<10} {n:>7,}  ({n/len(df)*100:5.1f}%)")

    print("\n  クラス別 (上位20):")
    for code, n in df["class_code"].value_counts().head(20).items():
        print(f"    {code:<8} {n:>7,}")

    print("\n  場別 A1..D2 抽出率:")
    for baba, g in df.groupby("baba_code"):
        r = (g["rule"] == "AB123").mean() * 100
        print(f"    {BABA.get(int(baba), str(baba)):<4}({int(baba):>2}) "
              f"{r:5.1f}%   n={len(g):,}")

    um = df[df["rule"] == "unmatched"]
    if len(um):
        print(f"\n  未マッチのレース名 (上位25 / 全{len(um):,}件):")
        for v, n in um["race_name"].value_counts().head(25).items():
            print(f"    {n:>5,}  {v!r}")


# =====================================================================
# D. パーサの取りこぼし
# =====================================================================
def parser_health(con) -> None:
    print("\n" + "=" * 64)
    print("■ D. パーサ健全性")
    ne = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    if not ne:
        return
    for col, note in [
        ("weight_carried", "斤量。JOCKEY_RE が外れると NULL になる"),
        ("trainer_name",   "同上"),
        ("transfer_from",  "所属[大井] 等。同上"),
        ("horse_id",       "血統番号。名寄せの生命線"),
    ]:
        n = con.execute(
            f"SELECT COUNT({col}) FROM entries").fetchone()[0]
        flag = "" if n / ne > 0.98 else "   ← 要調査"
        print(f"  {col:<16} {n:>9,}  ({n/ne*100:5.1f}%){flag}")

    # ROW_RE は「(N人気)」が無い行を落とす。中止/除外馬が消えている疑い
    print("\n  n_starters と results 行数の一致:")
    bad = con.execute("""
        SELECT COUNT(*) FROM (
          SELECT r.race_id, r.n_starters, COUNT(res.horse_no) c
          FROM races r JOIN results res USING(race_id)
          GROUP BY r.race_id HAVING c <> r.n_starters)
    """).fetchone()[0]
    print(f"    不一致レース {bad:,}")

    print("\n  1レースあたり出走頭数の分布:")
    d = pd.read_sql_query("SELECT n_starters FROM races", con)["n_starters"]
    print(f"    min {d.min()} / 中央 {d.median():.0f} / max {d.max()}")
    small = (d <= 5).sum()
    if small:
        print(f"    !! 5頭以下が {small:,}レース。"
              "取りこぼしなら Detail 版でないページを踏んでいる")


# =====================================================================
# E. コーナー通過順が「そもそもHTMLに在るか」
#    SQLでは分からない。1本だけ実際に取得して確かめる。
# =====================================================================
def probe_corner_source(con, fetch: bool = True) -> None:
    print("\n" + "=" * 64)
    print("■ E. コーナー通過順の在処")
    n = con.execute(
        "SELECT COUNT(corner_4) FROM results").fetchone()[0]
    if n:
        print(f"  corner_4 は {n:,}件入っている。probe不要")
        return
    print("  corner_4 は全NULL (fetchers.py が None 固定なので当然)")
    if not fetch:
        print("  --no-fetch 指定のため probe をスキップ")
        return

    row = con.execute("""
        SELECT race_date, baba_code, race_no, race_id
        FROM races ORDER BY race_date DESC LIMIT 1
    """).fetchone()
    if not row:
        return
    import datetime as dt
    from . import fetchers as F
    d = dt.date.fromisoformat(row[0])
    baba, rno, rid = int(row[1]), int(row[2]), row[3]
    url = F.url_result(d, baba, rno)
    print(f"  probe: {rid}  {url}")

    try:
        html = F.Fetcher(min_interval=1.0).get(url)
    except Exception as e:                     # noqa: BLE001
        print(f"  ✗ 取得失敗: {e}")
        return

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [l for l in text.split("\n") if l.strip()]

    hits = [(i, l) for i, l in enumerate(lines)
            if re.search(r"コーナー|通過|ハロン|ラップ|払戻|人気", l)]
    print(f"  HTML {len(html):,} bytes / 本文 {len(lines)} 行")
    print("  キーワード該当行とその直後2行:")
    shown = 0
    for i, l in hits:
        if "人気" in l and "コーナー" not in l:
            continue
        print(f"    [{i}] {l!r}")
        for j in range(i + 1, min(i + 3, len(lines))):
            print(f"        {lines[j]!r}")
        shown += 1
        if shown >= 8:
            break
    if not shown:
        print("    !! コーナー/通過/ラップ の記載なし。"
              "→ 成績ページには存在しない。別ページか、諦めて脚質は"
              "  タイムと上がり3Fから推定する方向に切り替える")

    # race_name 抽出行のずれ確認用に、周辺を生で出す
    print("\n  参考: 本文冒頭20行 (race_name 抽出ロジックの検証用)")
    for i, l in enumerate(lines[:20]):
        print(f"    [{i}] {l!r}")


# =====================================================================
def data_report(con, fetch: bool = True) -> None:
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

    # ---- ここから追加した欠損診断 ----
    null_report(con)
    race_name_report(con)
    parser_health(con)
    probe_corner_source(con, fetch=fetch)

    print("\n" + "=" * 64)
    print("診断おわり。C の A1..D2 抽出率が 6割を超えていれば "
          "features.py にクラスを入れる価値がある。")


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

    print("\n=== 人気順位別の推定精度 (1着馬) ===")
    print(f"  {'人気':>4} {'n':>7} {'実オッズ中央':>12} {'推定中央':>10} {'|誤差|中央':>10}")
    for pop, g in m[m["odds_source"] != "actual"].groupby("final_popularity"):
        if len(g) < 30 or pop > 10:
            continue
        print(f"  {int(pop):>4} {len(g):>7,} "
              f"{g['odds_true'].median():>12.1f} "
              f"{g['odds_est'].median():>10.1f} "
              f"{g['rel_err'].abs().median()*100:>9.1f}%")

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
    ap.add_argument("--no-fetch", action="store_true",
                    help="E のHTML probe を行わない(完全オフライン)")
    a = ap.parse_args()
    con = connect()
    if a.report or not (a.report or a.calibrate):
        data_report(con, fetch=not a.no_fetch)
    if a.calibrate:
        calibrate(con)


if __name__ == "__main__":
    main()
