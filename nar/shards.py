"""
シャードDBの統合と、開催日カレンダーの先読み。

merge:
  matrix で並列収集した shard-*.db を1本にまとめる。
  ATTACH + INSERT OR IGNORE。何度流しても冪等。

calendar:
  日別の開催一覧を1回叩いて、実在するレースだけキューに積む。
  総当たりだと約7割が非開催の404になる。ここを削るとリクエストが1/3。
"""
from __future__ import annotations

import argparse
import re
import datetime as dt
from pathlib import Path

from .core import JST, BABA, EXCLUDE_BABA, connect, DB_PATH, make_race_id
from .fetchers import Fetcher, ParseError, dump

# 統合対象テーブル(順序が外部キー依存なので races が先)
MERGE_TABLES = ["races", "entries", "results", "payouts",
                "odds_snapshots", "race_queue", "fetch_log"]


def merge(shard_paths: list[Path], out: Path = DB_PATH) -> None:
    con = connect(out)
    total = {t: 0 for t in MERGE_TABLES}

    for i, sp in enumerate(shard_paths):
        if not sp.exists():
            print(f"  skip (missing): {sp}")
            continue
        con.execute(f"ATTACH DATABASE ? AS s{i}", (str(sp),))
        for t in MERGE_TABLES:
            try:
                before = con.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
                # race_queue は done を優先したいので、後述の後処理で整合を取る
                con.execute(f"INSERT OR IGNORE INTO main.{t} SELECT * FROM s{i}.{t}")
                after = con.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
                total[t] += after - before
            except Exception as e:
                print(f"    ! {t}: {e}")
        con.commit()
        con.execute(f"DETACH DATABASE s{i}")
        print(f"  merged {sp.name}")

    # race_queue の整合: どこか1シャードで done なら done に寄せる
    for i, sp in enumerate(shard_paths):
        if not sp.exists():
            continue
        con.execute(f"ATTACH DATABASE ? AS s{i}", (str(sp),))
        con.execute(f"""
            UPDATE main.race_queue
            SET status = 'done'
            WHERE race_id IN (SELECT race_id FROM s{i}.race_queue WHERE status='done')
        """)
        con.execute(f"""
            UPDATE main.race_queue
            SET status = 'skipped'
            WHERE status != 'done'
              AND race_id IN (SELECT race_id FROM s{i}.race_queue WHERE status='skipped')
        """)
        con.commit()
        con.execute(f"DETACH DATABASE s{i}")

    print("\n=== merged rows (new) ===")
    for t, n in total.items():
        print(f"  {t:16s} +{n:,}")

    con.execute("VACUUM")
    size_mb = out.stat().st_size / 1e6
    print(f"\n  {out}  {size_mb:.1f} MB")


# =====================================================================
# 開催日カレンダー先読み
# =====================================================================
# 日別の開催一覧。ここから実在レースのリンクだけ拾う。
CAL_URL = "https://nar.netkeiba.com/top/race_list.html?kaisai_date={ymd}"
RACE_ID_RE = re.compile(r"race_id=(\d{12})")


def parse_race_ids(html: str, ymd: str) -> list[tuple[int, int]]:
    """
    一覧HTMLから (baba_code, race_no) を抽出。
    netkeiba の race_id は YYYY + baba(2) + MM + DD + RR。
    """
    found = set()
    for rid in RACE_ID_RE.findall(html):
        baba = int(rid[4:6])
        rno = int(rid[10:12])
        if baba in BABA and baba not in EXCLUDE_BABA and 1 <= rno <= 12:
            found.add((baba, rno))
    if not found:
        raise ParseError(f"開催一覧からレースを抽出できない ({ymd})",
                         dump(html, f"cal-{ymd}"))
    return sorted(found)


def enqueue_by_calendar(con, start: dt.date, end: dt.date,
                        f: Fetcher | None = None) -> int:
    """
    総当たりの代わりに、日別一覧で実在レースだけ積む。
    1日1リクエスト増えるが、404の空撃ちが7割消えるので大幅に得。
    """
    f = f or Fetcher(min_interval=2.0)
    from .backfill import NK_URL

    rows, day, n_days_hit = [], start, 0
    while day <= end:
        ymd = f"{day:%Y%m%d}"
        try:
            html = f.get(CAL_URL.format(ymd=ymd))
            pairs = parse_race_ids(html, ymd)
            n_days_hit += 1
            for baba, rno in pairs:
                rows.append({
                    "race_id": make_race_id(day, baba, rno),
                    "race_date": day.isoformat(),
                    "baba_code": baba, "race_no": rno,
                    "source_url": NK_URL.format(y=day.year, b=baba,
                                                m=day.month, d=day.day, r=rno),
                    "status": "pending", "attempts": 0, "last_error": None,
                    "updated_at": dt.datetime.now(JST).isoformat(),
                })
            print(f"  {ymd}: {len(pairs)}R")
        except ParseError as e:
            print(f"  {ymd}: 非開催 or 取得失敗 ({e})")
        day += dt.timedelta(days=1)

    if rows:
        con.executemany("""INSERT OR IGNORE INTO race_queue
            (race_id,race_date,baba_code,race_no,source_url,
             status,attempts,last_error,updated_at)
            VALUES (:race_id,:race_date,:baba_code,:race_no,:source_url,
                    :status,:attempts,:last_error,:updated_at)""", rows)
        con.commit()
    print(f"\n開催日 {n_days_hit}日 / 候補 {len(rows):,}R")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge")
    m.add_argument("paths", nargs="+", help="shard-*.db")
    m.add_argument("--out", default=str(DB_PATH))

    c = sub.add_parser("calendar")
    c.add_argument("--since", required=True)
    c.add_argument("--until", required=True)

    a = ap.parse_args()
    if a.cmd == "merge":
        merge([Path(p) for p in a.paths], Path(a.out))
    else:
        con = connect()
        enqueue_by_calendar(con,
                            dt.date.fromisoformat(a.since),
                            dt.date.fromisoformat(a.until))


if __name__ == "__main__":
    main()
