"""
NAR コレクタ中核モジュール

責務:
  - 場コード / race_id の正規化
  - SQLite への冪等な書き込み (何度流しても壊れない)
  - スナップショット時刻の決定ロジック

HTMLパース部は fetchers.py 側。ここは通信しない。
"""
from __future__ import annotations

import sqlite3
import datetime as dt
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nar.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# ---------------------------------------------------------------------
# 場コード (NAR公式 babaCode)
#   ※ 稼働前に1場ずつ実URLで要検証。コード体系は変わらないが確認は必須
# ---------------------------------------------------------------------
BABA: dict[int, str] = {
    30: "門別",
    31: "帯広ば",   # ばんえい。競走形態が全く違うので原則除外推奨
    35: "盛岡",
    36: "水沢",
    42: "浦和",
    43: "船橋",
    44: "大井",
    45: "川崎",
    46: "金沢",
    47: "笠松",
    48: "名古屋",
    50: "園田",
    51: "姫路",
    54: "高知",
    55: "佐賀",
}

# ばんえいは別競技として扱う。混ぜるとモデルが壊れる
EXCLUDE_BABA = {31}

# 券種の正規名
BET_TYPES = (
    "win",       # 単勝
    "place",     # 複勝
    "quinella",  # 馬連
    "exacta",    # 馬単
    "wide",      # ワイド
    "trio",      # 三連複
    "trifecta",  # 三連単
)

# クラスの強さ正規化 (小さいほど上位)。地区ごとの呼称ゆれをここで吸収する
CLASS_RANK = {
    "JpnI": 0, "JpnII": 1, "JpnIII": 2, "重賞": 3,
    "A1": 10, "A2": 11, "A3": 12,
    "B1": 20, "B2": 21, "B3": 22,
    "C1": 30, "C2": 31, "C3": 32,
    "D1": 40, "D2": 41,
}

TRACK_COND_NUM = {"良": 1, "稍重": 2, "稍": 2, "重": 3, "不良": 4, "不": 4}


# =====================================================================
# race_id
# =====================================================================
def make_race_id(date: dt.date, baba_code: int, race_no: int) -> str:
    return f"{date:%Y%m%d}-{baba_code:02d}-{race_no:02d}"


def parse_race_id(race_id: str) -> tuple[dt.date, int, int]:
    d, b, r = race_id.split("-")
    return dt.date(int(d[:4]), int(d[4:6]), int(d[6:8])), int(b), int(r)


# =====================================================================
# スナップショット計画
# =====================================================================
SNAPSHOT_PLAN = [
    ("morning", 240),  # 発走240分前 … 朝の薄いオッズ
    ("pre60",    60),
    ("pre15",    15),
    ("pre02",     2),  # ★ EV計算の基準。実際に賭けられる最後のオッズ
]


@dataclass(frozen=True)
class SnapshotTask:
    race_id: str
    post_at: dt.datetime
    tag: str
    fire_at: dt.datetime

    @property
    def min_to_post(self) -> int:
        return int((self.post_at - self.fire_at).total_seconds() // 60)


def plan_snapshots(race_id: str, post_time: str, date: dt.date) -> list[SnapshotTask]:
    """post_time は 'HH:MM'。発走時刻から逆算してスナップショット時刻を決める。"""
    hh, mm = (int(x) for x in post_time.split(":"))
    post_at = dt.datetime(date.year, date.month, date.day, hh, mm, tzinfo=JST)
    return [
        SnapshotTask(race_id, post_at, tag, post_at - dt.timedelta(minutes=lead))
        for tag, lead in SNAPSHOT_PLAN
    ]


def due_now(tasks: list[SnapshotTask], now: dt.datetime, window_min: int = 5
            ) -> list[SnapshotTask]:
    """cron が window_min 間隔で回る前提。発火予定を過ぎ、かつ締切前のものを返す。"""
    out = []
    for t in tasks:
        if t.fire_at <= now < t.fire_at + dt.timedelta(minutes=window_min):
            if now < t.post_at:          # 締切後に取っても無意味
                out.append(t)
    return out


# =====================================================================
# DB
# =====================================================================
def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return con


def upsert(con: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """全カラム上書きの冪等 UPSERT。再実行しても重複しない。"""
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join("?" * len(cols))
    collist = ",".join(cols)
    sql = f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})"
    con.executemany(sql, [[r[c] for c in cols] for r in rows])
    con.commit()
    return len(rows)


def log_fetch(con, target: str, kind: str, status: str, detail: str = "") -> None:
    con.execute(
        "INSERT INTO fetch_log (target, kind, status, detail, logged_at) VALUES (?,?,?,?,?)",
        (target, kind, status, detail, dt.datetime.now(JST).isoformat()),
    )
    con.commit()


# =====================================================================
# 派生特徴 — 「荒れる」構造要因の数値化
# =====================================================================
def odds_drift(con: sqlite3.Connection, race_id: str) -> dict[int, float]:
    """
    朝一 → 締切2分前 の単勝オッズ変化率を馬番ごとに返す。
    負値 = 買われて沈んだ(=資金流入)。
    これが極端な馬が居るレースは、良くも悪くも市場が何かを織り込んでいる。
    因果は特定できない。特徴量として置くだけ。
    """
    rows = con.execute("""
        SELECT combination, snapshot_tag, odds
        FROM odds_snapshots
        WHERE race_id = ? AND bet_type = 'win'
          AND snapshot_tag IN ('morning', 'pre02')
    """, (race_id,)).fetchall()

    m: dict[int, dict[str, float]] = {}
    for r in rows:
        m.setdefault(int(r["combination"]), {})[r["snapshot_tag"]] = r["odds"]

    return {
        no: (v["pre02"] - v["morning"]) / v["morning"]
        for no, v in m.items()
        if v.get("morning") and v.get("pre02")
    }


def front_runner_congestion(con: sqlite3.Connection, race_id: str) -> dict:
    """
    ハナ争い指数。前走4角を1〜2番手で通過した馬が何頭いるか、
    かつそれが外枠に偏っているか。
    先行馬が多く外に集中しているほど共倒れ確率が上がる = 荒れる。
    """
    rows = con.execute("""
        SELECT e.horse_no, e.last_corner4, r.n_starters
        FROM entries e JOIN races r USING (race_id)
        WHERE e.race_id = ?
    """, (race_id,)).fetchall()
    if not rows:
        return {}

    n = rows[0]["n_starters"]
    front = [r["horse_no"] for r in rows if r["last_corner4"] and r["last_corner4"] <= 2]
    if not front:
        return {"n_front": 0, "front_ratio": 0.0, "front_outer_bias": 0.0}

    return {
        "n_front": len(front),
        "front_ratio": len(front) / n,
        # 0=全部内枠 1=全部外枠
        "front_outer_bias": sum((h - 1) / max(n - 1, 1) for h in front) / len(front),
    }


def implied_prob(odds_map: dict[str, float], takeout: float | None = None
                 ) -> dict[str, float]:
    """
    オッズ → 市場の暗黙確率。
    素の 1/odds は控除率のぶん合計が1を超えるので、正規化して戻す。
    takeout を渡さなければ実測の overround から逆算する(推奨)。
    """
    raw = {k: 1.0 / v for k, v in odds_map.items() if v and v > 0}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}
