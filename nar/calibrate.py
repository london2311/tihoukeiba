"""
校正/疎通確認ツール。1レースだけ実際に取得して全パーサを通す。

    python -m nar.calibrate --date 2026-08-12 --baba 20 --race 9
"""
from __future__ import annotations

import argparse
import datetime as dt

from .core import BABA, make_race_id
from . import fetchers as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--baba", type=int, required=True,
                    help=" ".join(f"{k}={v}" for k, v in BABA.items()))
    ap.add_argument("--race", type=int, default=11)
    ap.add_argument("--odds", default="win,place,trifecta",
                    help="確認する券種をカンマ区切り")
    a = ap.parse_args()

    d = dt.date.fromisoformat(a.date)
    rid = make_race_id(d, a.baba, a.race)
    f = F.Fetcher(min_interval=2.0)

    # --- 開催確認 ---
    print(f"=== 開催一覧: {d} {BABA.get(a.baba, a.baba)} ===")
    lst = F.parse_race_list(f.get(F.url_race_list(d, a.baba)))
    print(f"  実施R: {lst or '(非開催)'}")
    if a.race not in lst:
        print(f"  !! {a.race}R は無い。上のRから選び直すこと")
        return

    # --- 成績 ---
    print(f"\n=== 成績 {rid} ===")
    print(f"  {F.url_result(d, a.baba, a.race)}")
    res = F.parse_result_page(
        f.get(F.url_result(d, a.baba, a.race)), d, a.baba, a.race, rid)
    r = res["race"]
    print(f"  {r['race_name']}")
    print(f"  {r['baba_name']} {r['race_no']}R {r['distance_m']}m "
          f"{r['surface']} {r['direction']} / {r['post_time']} / "
          f"{r['weather']} {r['track_cond']} / {r['n_starters']}頭")

    print(f"\n  {'着':>2} {'馬':>2} {'人':>2} {'馬名':<14} {'time':>6} "
          f"{'上3F':>5} {'騎手':<6} {'血統番号'}")
    for e, x in zip(res["entries"], res["results"]):
        print(f"  {x['finish_pos'] or '-':>2} {e['horse_no']:>2} "
              f"{x['final_popularity']:>2} {str(e['horse_name'])[:14]:<14} "
              f"{x['time_sec'] or '-':>6} {x['last_3f'] or '-':>5} "
              f"{str(e['jockey_name'])[:6]:<6} {e['horse_id']}")

    n_hid = sum(1 for e in res["entries"] if e["horse_id"])
    if n_hid < len(res["entries"]):
        print(f"\n  !! 血統番号が {n_hid}/{len(res['entries'])} しか取れない。"
              "馬の名寄せができず過去走特徴が作れなくなる")

    # --- オッズ ---
    now = dt.datetime.now().isoformat()
    for bet in [b.strip() for b in a.odds.split(",") if b.strip()]:
        print(f"\n=== オッズ({bet}) ===")
        try:
            rows = F.parse_odds_page(
                f.get(F.url_odds(d, a.baba, a.race, bet)), rid, bet, now)
        except F.ParseError as e:
            print(f"  ✗ {e}")
            continue
        print(f"  {len(rows)}件")
        for x in rows[:8]:
            up = f" - {x['odds_upper']}" if x["odds_upper"] else ""
            print(f"    {x['combination']:<12} {x['odds']}{up}")
        if len(rows) > 8:
            print(f"    ... 他 {len(rows)-8}件")
        if bet == "win" and rows:
            ov = sum(1 / x["odds"] for x in rows)
            print(f"  overround={ov:.4f} → 控除率 {(1-1/ov)*100:.1f}% "
                  f"{'(妥当)' if 0.15 < 1-1/ov < 0.30 else '(!! 異常)'}")

    # --- 払戻 ---
    print("\n=== 払戻 ===")
    try:
        pays = F.parse_payout_page(f.get(F.url_payout(d, a.baba, a.race)), rid)
        for p in pays:
            print(f"    {p['bet_type']:18s} {p['combination']:<12} "
                  f"{p['payout_yen']:>9,}円")
        if not pays:
            print("    !! 0件")
    except F.ParseError as e:
        print(f"  ✗ {e}")

    print("\n" + "=" * 60)
    print("校正完了。バックフィルを開始してよい。")


if __name__ == "__main__":
    main()
