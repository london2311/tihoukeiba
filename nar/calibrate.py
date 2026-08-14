"""
セレクタ校正ツール。稼働前に1回だけ実行する。

実ページを1レース取得して、
  - どのテーブルが結果表なのか
  - 実際のヘッダ名は何か
  - COLMAP に足りないマッピングはどれか
を機械的に洗い出し、そのまま貼れる形で出力する。

使い方:
    python -m nar.calibrate --date 2026-08-01 --baba 44 --race 11

    # ネットに繋がらない環境なら、保存済みHTMLでもよい
    python -m nar.calibrate --html saved.html --date 2026-08-01 --baba 44 --race 11
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from bs4 import BeautifulSoup

from .core import BABA
from .fetchers import (Fetcher, SEL, COLMAP, parse_race_page, parse_payouts,
                       _table_to_dicts, dump, ParseError, norm_header)
from .backfill import NK_URL, ENCODING

REQUIRED = {"finish_pos", "horse_no", "horse_name"}
WANTED = {"final_win_odds", "final_popularity", "bracket_no", "sex_age",
          "jockey_name", "corner_str", "time_str", "last_3f",
          "weight_carried", "body_weight_str", "trainer_name"}


def inspect(html: str, race_date: dt.date, baba: int, race_no: int) -> None:
    soup = BeautifulSoup(html, "html.parser")
    print(f"\n{'='*66}\nHTML {len(html):,} bytes")

    title = soup.select_one("h1")
    print(f"title: {title.get_text(strip=True) if title else '(なし)'}")

    # ---------- 全テーブルを列挙 ----------
    tables = soup.find_all("table")
    print(f"\n--- <table> が {len(tables)} 個 ---")
    best_i, best_score = None, -1
    for i, t in enumerate(tables):
        rows = t.find_all("tr")
        if not rows:
            continue
        hdr = [norm_header(c.get_text(strip=True))
               for c in rows[0].find_all(["th", "td"])]
        mapped = {COLMAP[h] for h in hdr if h in COLMAP}
        score = len(REQUIRED & mapped) * 10 + len(WANTED & mapped)
        cls = " ".join(t.get("class") or []) or "(class無し)"
        mark = ""
        if REQUIRED <= mapped:
            mark = "  ★結果表の可能性大"
        elif any(x in " ".join(hdr) for x in ("払戻", "単勝", "三連")):
            mark = "  ◆払戻表かも"
        print(f"  [{i}] class='{cls}' rows={len(rows)} score={score}{mark}")
        print(f"      headers: {hdr[:14]}")
        if score > best_score:
            best_i, best_score = i, score

    # 全テーブルのヘッダが空 = 中身の無い枠だけのページ。
    # 取得先ドメインが違う / 非開催 / JS描画 のいずれか。
    all_empty = all(
        not [c.get_text(strip=True)
             for c in (t.find_all("tr") or [None])[0].find_all(["th", "td"])]
        for t in tables if t.find_all("tr")
    ) if tables else True

    if best_i is None or all_empty:
        print("\n!! 中身のあるテーブルが無い。")
        print("   考えられる原因: 取得先ドメイン違い / 非開催 / JS描画")
        text = soup.get_text(" ", strip=True)
        print(f"\n--- ページ本文 冒頭600字 ---\n{text[:600]}")
        print(f"\n--- 本文の長さ: {len(text):,}字 ---")
        if len(text) < 200:
            print("   → 本文がほぼ空。ドメイン違いの可能性が高い")
        return

    # ---------- 最有力テーブルの詳細 ----------
    t = tables[best_i]
    hdr = [norm_header(c.get_text(strip=True))
           for c in t.find_all("tr")[0].find_all(["th", "td"])]
    print(f"\n--- 最有力: table[{best_i}] ---")

    unmapped = [h for h in hdr if h not in COLMAP]
    mapped = {COLMAP[h] for h in hdr if h in COLMAP}

    if unmapped:
        print("\n■ COLMAP に未登録のヘッダ（下記を fetchers.py の COLMAP に追記）:")
        for h in unmapped:
            print(f'    "{h}": "???",')
    else:
        print("\n■ 全ヘッダがマップ済み")

    miss_req = REQUIRED - mapped
    if miss_req:
        print(f"\n!! 必須が取れない: {miss_req}  ← 上の ??? を埋めること")
    miss_want = WANTED - mapped
    if miss_want:
        print(f"\n□ 任意で未取得: {sorted(miss_want)}")

    cls = ".".join(t.get("class") or [])
    if cls:
        cur = SEL["nk_result_tbl"]
        if cls not in cur:
            print(f"\n■ SEL['nk_result_tbl'] に 'table.{cls}' を追加すること")
            print(f'    現在: "{cur}"')
        else:
            print(f"\n■ SEL['nk_result_tbl'] はこのテーブルに一致済み (table.{cls})")

    # ---------- 1行サンプル ----------
    rows = _table_to_dicts(t)
    if rows:
        print("\n--- 1行目のパース結果 ---")
        for k, v in rows[0].items():
            if not k.endswith("_href"):
                print(f"    {k:20s} = {v!r}")
        hrefs = {k: v for k, v in rows[0].items() if k.endswith("_href")}
        if hrefs:
            print("\n    リンク（horse_id/jockey_id の抽出元）:")
            for k, v in hrefs.items():
                print(f"      {k:24s} {v}")
        else:
            print("\n    !! <a href> が無い → horse_id が取れない。"
                  "馬名の名寄せができず過去走特徴が作れなくなる。")

    # ---------- 払戻 ----------
    pays = parse_payouts(soup, "TEST")
    print(f"\n--- 払戻: {len(pays)}件 ---")
    for p in pays[:8]:
        print(f"    {p['bet_type']:10s} {p['combination']:12s} {p['payout_yen']:>8,}円")
    if not pays:
        print("    !! 0件。SEL['nk_payout_tbl'] を上の ◆ のテーブルに合わせること")

    # ---------- 本番パーサを通す ----------
    print(f"\n{'='*66}\n本番 parse_race_page() を実行:")
    try:
        d = parse_race_page(html, race_date, baba, race_no)
        r = d["race"]
        print("  ✓ 成功")
        print(f"    {r['baba_name']} {r['race_no']}R "
              f"{r['surface']} {r['distance_m']}m {r['direction']} "
              f"{r['class_code']} 馬場{r['track_cond']} {r['n_starters']}頭")
        odds = [x["final_win_odds"] for x in d["results"]]
        got = sum(1 for o in odds if o)
        print(f"    確定単勝オッズ: {got}/{len(odds)}頭 取得")
        ids = sum(1 for e in d["entries"] if e["horse_id"])
        print(f"    horse_id:       {ids}/{len(d['entries'])}頭 取得")
        if got < len(odds) or ids < len(d["entries"]):
            print("    → 欠けがある。上の未登録ヘッダを埋めれば埋まるはず")
        else:
            print("\n  校正完了。バックフィルを開始してよい。")
    except ParseError as e:
        print(f"  ✗ 失敗: {e}")
        print("    → 上の ■ の指示どおり fetchers.py を直してから再実行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--baba", type=int, required=True,
                    help="場コード " + " ".join(f"{k}={v}" for k, v in BABA.items()))
    ap.add_argument("--race", type=int, default=11)
    ap.add_argument("--html", help="保存済みHTMLを使う(通信しない)")
    a = ap.parse_args()

    d = dt.date.fromisoformat(a.date)
    if a.html:
        html = Path(a.html).read_text(encoding="utf-8", errors="replace")
        print(f"local: {a.html}")
    else:
        url = NK_URL.format(y=d.year, b=a.baba, m=d.month, d=d.day, r=a.race)
        print(f"fetch: {url}")
        html = Fetcher(min_interval=2.0).get(url, encoding=ENCODING)
        print(f"saved: {dump(html, f'calib-{a.date}-{a.baba}-{a.race}')}")

    inspect(html, d, a.baba, a.race)


if __name__ == "__main__":
    main()
