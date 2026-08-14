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
import datetime as dt
from pathlib import Path

from .core import connect, DB_PATH

# 統合対象テーブル(順序が外部キー依存なので races が先)
MERGE_TABLES = ["races", "entries", "results", "payouts",
                "odds_snapshots", "race_queue", "probed", "fetch_log"]


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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge")
    m.add_argument("paths", nargs="+", help="shard-*.db")
    m.add_argument("--out", default=str(DB_PATH))

    a = ap.parse_args()
    merge([Path(p) for p in a.paths], Path(a.out))


if __name__ == "__main__":
    main()
