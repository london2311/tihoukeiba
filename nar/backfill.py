"""
バックフィル本体 — keiba.go.jp 版。

1レースあたりのリクエスト数を券種で選べるようにしてある。ここが総時間を決める。

  --odds win        : 成績 + 単勝        = 2 req/race  ← 既定。まずこれ
  --odds core       : + 複勝馬連馬単ワイド = 6 req/race
  --odds all        : + 三連複三連単      = 8 req/race

単勝から始める理由: 控除率が最も低く(実測21.3%)、データも完全に揃う。
三連単は市場を4割上回らないと損益分岐しない。勝ち目のある券種から手をつける。

所要時間の目安 (2秒間隔):
  win  → 4秒/race。10,000race = 11時間。6シャードで2時間弱
  all  → 16秒/race。10,000race = 44時間。6シャードで8時間
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from .core import (JST, BABA, EXCLUDE_BABA, connect, upsert, log_fetch,
                   make_race_id)
from . import fetchers as F

ODDS_SETS = {
    "win":  ["win"],
    "core": ["win", "place", "quinella", "exacta", "wide"],
    "all":  ["win", "place", "quinella", "exacta", "wide", "trio", "trifecta"],
    "none": [],
}


# =====================================================================
# キュー構築 — 日別レース一覧から実在レースだけ積む
# =====================================================================
def enqueue_by_calendar(con, f: F.Fetcher, start: dt.date, end: dt.date,
                        baba: list[int] | None = None) -> int:
    targets = baba or [b for b in BABA if b not in EXCLUDE_BABA]
    rows, day, hit_days = [], start, 0

    while day <= end:
        day_hit = False
        for b in targets:
            try:
                html = f.get(F.url_race_list(day, b))
                nos = F.parse_race_list(html)
            except F.ParseError:
                nos = []
            if not nos:
                continue
            day_hit = True
            for r in nos:
                rows.append({
                    "race_id": make_race_id(day, b, r),
                    "race_date": day.isoformat(),
                    "baba_code": b, "race_no": r,
                    "source_url": F.url_result(day, b, r),
                    "status": "pending", "attempts": 0, "last_error": None,
                    "updated_at": dt.datetime.now(JST).isoformat(),
                })
            print(f"  {day} {BABA[b]:4s} {len(nos)}R")
        hit_days += day_hit
        day += dt.timedelta(days=1)

    if rows:
        con.executemany("""INSERT OR IGNORE INTO race_queue
            (race_id,race_date,baba_code,race_no,source_url,
             status,attempts,last_error,updated_at)
            VALUES (:race_id,:race_date,:baba_code,:race_no,:source_url,
                    :status,:attempts,:last_error,:updated_at)""", rows)
        con.commit()
    print(f"\n開催日 {hit_days}日 / 候補 {len(rows):,}R")
    return len(rows)


def mark(con, race_id: str, status: str, err: str = "") -> None:
    con.execute("""UPDATE race_queue SET status=?, attempts=attempts+1,
                   last_error=?, updated_at=? WHERE race_id=?""",
                (status, err[:500], dt.datetime.now(JST).isoformat(), race_id))
    con.commit()


# =====================================================================
# 1レース分の取得
# =====================================================================
def fetch_one(con, f: F.Fetcher, row, bets: list[str], with_payout: bool) -> int:
    rid = row["race_id"]
    day = dt.date.fromisoformat(row["race_date"])
    baba, rno = row["baba_code"], row["race_no"]

    html = f.get(F.url_result(day, baba, rno))
    d = F.parse_result_page(html, day, baba, rno, rid)

    now = dt.datetime.now(JST).isoformat()
    odds_rows: list[dict] = []
    win_map: dict[int, float] = {}

    for bet in bets:
        try:
            oh = f.get(F.url_odds(day, baba, rno, bet))
            got = F.parse_odds_page(oh, rid, bet, now)
            odds_rows += got
            if bet == "win":
                win_map = {int(x["combination"]): x["odds"] for x in got}
        except F.ParseError as e:
            log_fetch(con, rid, f"odds:{bet}", "empty", str(e))

    # 確定単勝オッズを results に焼き込む(特徴量生成が楽になる)
    for r in d["results"]:
        r["final_win_odds"] = win_map.get(r["horse_no"])

    payouts = []
    if with_payout:
        try:
            payouts = F.parse_payout_page(
                f.get(F.url_payout(day, baba, rno)), rid)
        except F.ParseError:
            pass

    upsert(con, "races",   [d["race"]])
    upsert(con, "entries",  d["entries"])
    upsert(con, "results",  d["results"])
    if odds_rows:
        upsert(con, "odds_snapshots", odds_rows)
    if payouts:
        upsert(con, "payouts", payouts)

    n_odds = sum(1 for r in d["results"] if r["final_win_odds"])
    print(f"  ok  {rid} {d['race']['baba_name']:4s} "
          f"{d['race']['distance_m']}m {d['race']['n_starters']:>2}頭 "
          f"odds {n_odds}/{d['race']['n_starters']} "
          f"({len(odds_rows)}組)")
    return len(odds_rows)


def process(con, f: F.Fetcher, limit: int, bets: list[str], with_payout: bool,
            deadline: dt.datetime | None = None,
            shard: tuple[int, int] | None = None) -> dict:
    sql = ("SELECT * FROM race_queue WHERE status IN ('pending','retry') "
           "AND attempts < 3")
    args: list = []
    if shard:
        i, n = shard
        sql += " AND (rowid % ?) = ?"
        args += [n, i]
    sql += " ORDER BY race_date DESC, baba_code, race_no LIMIT ?"
    args.append(limit)

    stat = {"ok": 0, "error": 0, "seen": 0}
    for row in con.execute(sql, args).fetchall():
        if deadline and dt.datetime.now(JST) >= deadline:
            print("  deadline reached, stopping cleanly")
            break
        stat["seen"] += 1
        rid = row["race_id"]
        try:
            fetch_one(con, f, row, bets, with_payout)
            mark(con, rid, "done")
            log_fetch(con, rid, "race", "ok")
            stat["ok"] += 1
        except F.ParseError as e:
            msg = str(e)
            mark(con, rid, "skipped" if "404" in msg else "retry", msg)
            log_fetch(con, rid, "race", "parse_error", msg)
            stat["error"] += 1
            print(f"  ERR {rid}: {msg}", file=sys.stderr)
        except Exception as e:
            mark(con, rid, "retry", repr(e))
            stat["error"] += 1
            print(f"  ERR {rid}: {e!r}", file=sys.stderr)
    return stat


# =====================================================================
def report(con) -> None:
    print("\n===== coverage =====")
    for r in con.execute("SELECT status,COUNT(*) n FROM race_queue "
                         "GROUP BY status ORDER BY n DESC"):
        print(f"  queue {r['status']:10s} {r['n']:>7,}")

    nr = con.execute("SELECT COUNT(*) c FROM races").fetchone()["c"]
    ne = con.execute("SELECT COUNT(*) c FROM entries").fetchone()["c"]
    no = con.execute("SELECT COUNT(*) c FROM results "
                     "WHERE final_win_odds IS NOT NULL").fetchone()["c"]
    print(f"  races        {nr:>7,}")
    print(f"  entries      {ne:>7,}")
    if ne:
        print(f"  with odds    {no:>7,}  ({no/ne*100:.1f}%)")

    print("\n  odds by bet_type:")
    for r in con.execute("SELECT bet_type,COUNT(*) n FROM odds_snapshots "
                         "GROUP BY bet_type ORDER BY n DESC"):
        print(f"    {r['bet_type']:18s} {r['n']:>9,}")

    rng = con.execute("SELECT MIN(race_date) a,MAX(race_date) b "
                      "FROM races").fetchone()
    if rng["a"]:
        print(f"\n  date range   {rng['a']} .. {rng['b']}")
    print("\n  per venue:")
    for r in con.execute("SELECT baba_name,COUNT(*) n FROM races "
                         "GROUP BY baba_name ORDER BY n DESC"):
        print(f"    {r['baba_name']:6s} {r['n']:>6,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--baba", help="場コードをカンマ区切り")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--odds", choices=list(ODDS_SETS), default="win")
    ap.add_argument("--payout", action="store_true")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--shard", help="'i/n'")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    con = connect()
    if a.report:
        report(con); return

    f = F.Fetcher(min_interval=a.interval)

    if a.since:
        end = dt.date.fromisoformat(a.until) if a.until else \
            dt.date.today() - dt.timedelta(days=1)
        baba = [int(x) for x in a.baba.split(",")] if a.baba else None
        enqueue_by_calendar(con, f, dt.date.fromisoformat(a.since), end, baba)

    shard = None
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        shard = (i, n); print(f"shard {i}/{n}")

    deadline = (dt.datetime.now(JST) + dt.timedelta(hours=a.hours)) \
        if a.hours else None
    bets = ODDS_SETS[a.odds]
    print(f"odds set = {a.odds} {bets}  ({len(bets)+1+int(a.payout)} req/race)")

    stat = process(con, f, a.limit, bets, a.payout, deadline, shard)
    print(f"\nseen={stat['seen']} ok={stat['ok']} error={stat['error']}")
    report(con)


if __name__ == "__main__":
    main()
