"""
バックフィル本体。

戦略:
  1) 日付 × 場 × R を総当たりで race_queue に積む(存在しないものは404で自然に脱落)
  2) キューを status='pending' から順に消化
  3) 1件ごとに commit。落ちても続きから走る

10000レースの目安:
  地方は1日 全場合計で150〜250R。開催日ベースなら約60日ぶんで到達する。
  ただしモデル用には「同じ馬の過去走」が要るので、実際は2〜3年ぶん
  (約100,000〜150,000R)を取ることを推奨する。2秒間隔で60〜80時間。
  GitHub Actions を1日3ジョブ回せば2〜3週間で埋まる。急ぐ必要はない。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from itertools import product

from .core import (JST, BABA, EXCLUDE_BABA, connect, upsert, log_fetch,
                   make_race_id)
from .fetchers import Fetcher, parse_race_page, ParseError

# netkeiba 地方の race_id: YYYY + babaCode(2) + MM + DD + RR
NK_URL = "https://db.netkeiba.com/race/{y:04d}{b:02d}{m:02d}{d:02d}{r:02d}/"
MAX_R = 12


def enqueue_range(con, start: dt.date, end: dt.date,
                  baba: list[int] | None = None) -> int:
    """期間内の候補を全部積む。開催の有無は問わない(404で落とす方が単純で堅い)。"""
    targets = baba or [b for b in BABA if b not in EXCLUDE_BABA]
    rows, day = [], start
    while day <= end:
        for b, r in product(targets, range(1, MAX_R + 1)):
            rows.append({
                "race_id": make_race_id(day, b, r),
                "race_date": day.isoformat(),
                "baba_code": b, "race_no": r,
                "source_url": NK_URL.format(y=day.year, b=b, m=day.month,
                                            d=day.day, r=r),
                "status": "pending", "attempts": 0, "last_error": None,
                "updated_at": dt.datetime.now(JST).isoformat(),
            })
        day += dt.timedelta(days=1)

    # 既存行は上書きしない(done を pending に戻さない)
    cur = con.executemany(
        """INSERT OR IGNORE INTO race_queue
           (race_id, race_date, baba_code, race_no, source_url,
            status, attempts, last_error, updated_at)
           VALUES (:race_id,:race_date,:baba_code,:race_no,:source_url,
                   :status,:attempts,:last_error,:updated_at)""", rows)
    con.commit()
    return cur.rowcount


def mark(con, race_id: str, status: str, err: str = "") -> None:
    con.execute("""UPDATE race_queue
                   SET status=?, attempts=attempts+1, last_error=?, updated_at=?
                   WHERE race_id=?""",
                (status, err[:500], dt.datetime.now(JST).isoformat(), race_id))
    con.commit()


def process(con, f: Fetcher, limit: int, deadline: dt.datetime | None = None,
            shard: tuple[int, int] | None = None) -> dict:
    """
    shard=(i, n) で n 分割の i 番目だけを処理する。
    enqueue_range は決定的な順序で挿入するので rowid も全シャードで一致する。
    → 各ジョブが空DBから始めても、担当範囲が重複しない。
    """
    sql = """SELECT * FROM race_queue
             WHERE status IN ('pending','retry') AND attempts < 3"""
    args: list = []
    if shard:
        i, n = shard
        sql += " AND (rowid % ?) = ?"
        args += [n, i]
    sql += " ORDER BY race_date DESC, baba_code, race_no LIMIT ?"
    args.append(limit)
    q = con.execute(sql, args).fetchall()

    stat = {"ok": 0, "empty": 0, "error": 0, "seen": 0}
    for row in q:
        if deadline and dt.datetime.now(JST) >= deadline:
            print("  deadline reached, stopping cleanly")
            break
        rid = row["race_id"]
        stat["seen"] += 1
        try:
            html = f.get(row["source_url"], encoding="euc-jp")
            data = parse_race_page(
                html,
                dt.date.fromisoformat(row["race_date"]),
                row["baba_code"], row["race_no"], row["source_url"])

            upsert(con, "races",   [data["race"]])
            upsert(con, "entries",  data["entries"])
            upsert(con, "results",  data["results"])
            upsert(con, "payouts",  data["payouts"])
            mark(con, rid, "done")
            log_fetch(con, rid, "race", "ok",
                      f"{data['race']['n_starters']}頭")
            stat["ok"] += 1
            print(f"  ok  {rid} {data['race']['baba_name']} "
                  f"{data['race']['distance_m']}m {data['race']['n_starters']}頭")

        except ParseError as e:
            msg = str(e)
            if "404" in msg:
                mark(con, rid, "skipped", "no such race")   # 非開催。正常
                stat["empty"] += 1
            else:
                mark(con, rid, "retry", msg)
                log_fetch(con, rid, "race", "parse_error", msg)
                stat["error"] += 1
                print(f"  ERR {rid}: {msg}", file=sys.stderr)
        except Exception as e:
            mark(con, rid, "retry", repr(e))
            log_fetch(con, rid, "race", "http_error", repr(e))
            stat["error"] += 1
            print(f"  ERR {rid}: {e!r}", file=sys.stderr)
    return stat


def report(con) -> None:
    print("\n===== coverage =====")
    for r in con.execute("""SELECT status, COUNT(*) n FROM race_queue
                            GROUP BY status ORDER BY n DESC"""):
        print(f"  queue {r['status']:10s} {r['n']:>7,}")
    n_race = con.execute("SELECT COUNT(*) c FROM races").fetchone()["c"]
    n_ent = con.execute("SELECT COUNT(*) c FROM entries").fetchone()["c"]
    n_odds = con.execute(
        "SELECT COUNT(*) c FROM results WHERE final_win_odds IS NOT NULL"
    ).fetchone()["c"]
    print(f"  races        {n_race:>7,}")
    print(f"  entries      {n_ent:>7,}")
    print(f"  with odds    {n_odds:>7,}  "
          f"({n_odds / n_ent * 100:.1f}% of entries)" if n_ent else "")

    rng = con.execute(
        "SELECT MIN(race_date) a, MAX(race_date) b FROM races").fetchone()
    if rng["a"]:
        print(f"  date range   {rng['a']} .. {rng['b']}")

    print("\n  per venue:")
    for r in con.execute("""SELECT baba_name, COUNT(*) n FROM races
                            GROUP BY baba_name ORDER BY n DESC"""):
        print(f"    {r['baba_name']:6s} {r['n']:>6,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD")
    ap.add_argument("--until", help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--baba", help="場コードをカンマ区切りで絞る")
    ap.add_argument("--limit", type=int, default=500, help="今回処理する件数")
    ap.add_argument("--interval", type=float, default=2.0, help="秒。2未満禁止")
    ap.add_argument("--hours", type=float, default=None,
                    help="この時間で打ち切る(Actions の6h制限用)")
    ap.add_argument("--shard", help="'i/n' 形式。例 0/6")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    con = connect()
    if a.report:
        report(con)
        return

    if a.since:
        end = dt.date.fromisoformat(a.until) if a.until else \
            dt.date.today() - dt.timedelta(days=1)
        baba = [int(x) for x in a.baba.split(",")] if a.baba else None
        n = enqueue_range(con, dt.date.fromisoformat(a.since), end, baba)
        print(f"enqueued {n:,} candidates")

    deadline = (dt.datetime.now(JST) + dt.timedelta(hours=a.hours)) if a.hours else None
    f = Fetcher(min_interval=max(a.interval, 2.0))
    shard = None
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= i < n, "--shard は 0/n .. (n-1)/n"
        shard = (i, n)
        print(f"shard {i}/{n}")
    stat = process(con, f, a.limit, deadline, shard)
    print(f"\nseen={stat['seen']} ok={stat['ok']} "
          f"skipped={stat['empty']} error={stat['error']}")
    report(con)


if __name__ == "__main__":
    main()
