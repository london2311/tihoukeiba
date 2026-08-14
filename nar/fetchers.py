"""
取得層 — keiba.go.jp (NAR公式) 版。

netkeiba を捨てて全面移行した理由:
  - nar.netkeiba.com は結果行を JS で描画する。ヘッダしか HTML に無い
  - keiba.go.jp はサーバー側で完全な HTML を返す
  - 血統登録番号 (k_lineageLoginCode) が取れる = 馬の名寄せができる
  - ★過去レースの最終オッズが全券種そろって残っている★
    → 「三連系の過去オッズは取れない」という当初の前提は誤りだった。
      前方スナップショット収集は不要。

実データで確認済みの形式 (2026-08-12 大井9R):
  成績行: "3枠3番 (1人気)ブルーヒサシン 1:11.6 ［36.9］藤田凌（54.0）立花伸［大井］"
  条件:   "9R / 20:10発走 / 特 / 1200ダ(右) / 曇 / 不良"
  オッズ: 馬番 | 馬名 | オッズ の素直な表
"""
from __future__ import annotations

import re
import time
import random
import datetime as dt
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

DUMP_DIR = Path(__file__).resolve().parent.parent / "data" / "dumps"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

BASE = "https://sp.keiba.go.jp/KeibaWebSP_IPAT/TodayRaceInfo"

# ---------------------------------------------------------------------
# ★NAR公式の場コードは netkeiba と別体系★
#   netkeiba: 大井=44   NAR公式: 大井=20
# ---------------------------------------------------------------------
NAR_BABA: dict[int, str] = {
    3: "門別", 4: "帯広ば",
    10: "盛岡", 11: "水沢",
    18: "浦和", 19: "船橋", 20: "大井", 21: "川崎",
    22: "金沢", 23: "笠松", 24: "名古屋",
    27: "園田", 28: "姫路",
    31: "高知", 32: "佐賀",
}
EXCLUDE_BABA = {4}   # ばんえいは別競技

BET_PAGES = {
    "win":      "S_OddsTan_ipat",
    "place":    "S_OddsFuku_ipat",
    "quinella": "S_OddsUmLenFuku_ipat",
    "exacta":   "S_OddsUmLenTan_ipat",
    "wide":     "S_OddsWide_ipat",
    "trio":     "S_Odds3LenFuku_ipat",
    "trifecta": "S_Odds3LenTan_ipat",
}


class ParseError(Exception):
    def __init__(self, msg: str, dump_path: Path | None = None):
        super().__init__(msg + (f"  [dump: {dump_path}]" if dump_path else ""))
        self.dump_path = dump_path


# =====================================================================
# URL
# =====================================================================
def _d(date: dt.date) -> str:
    return quote(f"{date:%Y/%m/%d}", safe="")


def url_race_list(date: dt.date, baba: int) -> str:
    return f"{BASE}/S_RaceList_ipat?k_raceDate={_d(date)}&k_babaCode={baba}"


def url_result(date: dt.date, baba: int, race_no: int) -> str:
    """全頭表示のほう。通常版は上位5頭しか出さないので必ずこちらを使う。"""
    return (f"{BASE}/S_RaceMarkTableDetail_ipat?k_raceDate={_d(date)}"
            f"&k_raceNo={race_no}&k_babaCode={baba}")


def url_odds(date: dt.date, baba: int, race_no: int, bet: str) -> str:
    return (f"{BASE}/{BET_PAGES[bet]}?k_raceDate={_d(date)}"
            f"&k_raceNo={race_no}&k_babaCode={baba}")


def url_payout(date: dt.date, baba: int, race_no: int) -> str:
    return (f"{BASE}/S_RefundMoneyList_ipat?k_raceDate={_d(date)}"
            f"&k_raceNo={race_no}&k_babaCode={baba}")


# =====================================================================
# HTTP
# =====================================================================
class Fetcher:
    def __init__(self, min_interval: float = 2.0, jitter: float = 0.8,
                 timeout: float = 20.0, max_retry: int = 4):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "ja"})
        self.min_interval = max(min_interval, 1.5)
        self.jitter = jitter
        self.timeout = timeout
        self.max_retry = max_retry
        self._last = 0.0

    def _wait(self):
        sleep = self.min_interval - (time.monotonic() - self._last) \
            + random.uniform(0, self.jitter)
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.monotonic()

    def get(self, url: str) -> str:
        last = None
        for i in range(self.max_retry):
            self._wait()
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    raise ParseError(f"404 {url}")
                if r.status_code in (429, 503):
                    time.sleep(30 * (2 ** i))
                    continue
                r.raise_for_status()
                r.encoding = "utf-8"
                return r.text
            except requests.RequestException as e:
                last = e
                time.sleep(3 * (2 ** i))
        raise ParseError(f"fetch failed: {url} ({last})")


def dump(html: str, tag: str) -> Path:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    p = DUMP_DIR / f"{tag}-{dt.datetime.now():%Y%m%d%H%M%S}.html"
    p.write_text(html, encoding="utf-8")
    return p


def _txt(soup) -> str:
    return soup.get_text("\n", strip=True)


# =====================================================================
# 成績行のパース
#
#   "3枠3番 (1人気)ブルーヒサシン 1:11.6 ［36.9］藤田凌（54.0）立花伸［大井］"
#   "5枠5番 (2人気)フルム 1:11.7 ［36.9］ クビ 和田譲（56.0）渡邉和［大井］"
#
#   1着には着差が無い。レーティングが入る場合もある。
#   → 固定位置ではなくアンカー記号から削り取る方式にする。
# =====================================================================
ROW_RE = re.compile(
    r"(?P<waku>\d+)\s*枠\s*(?P<uma>\d+)\s*番"
    r"\s*[\(（]\s*(?P<pop>\d+)\s*人気\s*[\)）]"
    r"(?P<rest>.*)$",
    re.S,
)
TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2})\.(\d)")
AGARI_RE = re.compile(r"[［\[]\s*(\d+\.\d)\s*[］\]]")
JOCKEY_RE = re.compile(
    r"(?P<jockey>[^\s（(]+)\s*[（(]\s*(?P<kin>\d+\.?\d*)\s*[）)]"
    r"\s*(?P<trainer>[^\s［\[]+)\s*[［\[]\s*(?P<belong>[^］\]]+)"
)


def parse_result_row(pos_text: str, row_text: str, race_id: str) -> dict | None:
    m = ROW_RE.search(row_text)
    if not m:
        return None
    rest = m.group("rest")

    ag = AGARI_RE.search(rest)
    last_3f = float(ag.group(1)) if ag else None

    head = rest[: ag.start()] if ag else rest
    tm = TIME_RE.search(head)
    if tm:
        mi, sec, dec = tm.group(1), tm.group(2), tm.group(3)
        time_sec = (int(mi) * 60 if mi else 0) + int(sec) + int(dec) / 10
        horse_name = head[: tm.start()].strip()
    else:
        time_sec = None
        horse_name = head.strip()

    tail = rest[ag.end():] if ag else rest
    jm = JOCKEY_RE.search(tail)
    jockey = kin = trainer = belong = margin = None
    if jm:
        jockey = jm.group("jockey")
        kin = float(jm.group("kin"))
        trainer = jm.group("trainer")
        belong = jm.group("belong")
        margin = tail[: jm.start()].strip() or None

    fm = re.search(r"\d+", pos_text)
    return {
        "race_id": race_id,
        "horse_no": int(m.group("uma")),
        "bracket_no": int(m.group("waku")),
        "horse_name": horse_name or None,
        "final_popularity": int(m.group("pop")),
        "finish_pos": int(fm.group()) if fm else None,
        "time_sec": time_sec,
        "last_3f": last_3f,
        "margin": margin,
        "jockey_name": jockey,
        "weight_carried": kin,
        "trainer_name": trainer,
        "belong": belong,
    }


COND_DIST = re.compile(r"(\d{3,4})\s*(ダ|芝)\s*[\(（]\s*(右|左|直)")
COND_TIME = re.compile(r"(\d{1,2}):(\d{2})\s*発走")
COND_TRACK = re.compile(r"^(良|稍重|重|不良)$", re.M)
COND_WEATHER = re.compile(r"^(晴|曇|雨|小雨|雪|小雪)$", re.M)


def parse_result_page(html: str, race_date: dt.date, baba: int,
                      race_no: int, race_id: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    text = _txt(soup)

    if "競走成績" not in text:
        raise ParseError("成績ページではない", dump(html, f"notresult-{race_id}"))

    dm = COND_DIST.search(text)
    if not dm:
        raise ParseError("距離/馬場が取れない", dump(html, f"nocond-{race_id}"))
    distance = int(dm.group(1))
    surface = "dirt" if dm.group(2) == "ダ" else "turf"
    direction = {"右": "right", "左": "left", "直": "straight"}[dm.group(3)]

    tm = COND_TIME.search(text)
    post_time = f"{int(tm.group(1)):02d}:{tm.group(2)}" if tm else None
    trk = COND_TRACK.search(text)
    wea = COND_WEATHER.search(text)

    race_name = None
    lines = [l for l in text.split("\n") if l.strip()]
    for i, l in enumerate(lines):
        if "競馬場" in l and str(race_date.year) in l and i + 1 < len(lines):
            race_name = lines[i + 1].strip()
            break

    entries, results = [], []
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        pos = cells[0].get_text(" ", strip=True)
        body = cells[1].get_text(" ", strip=True)
        if not re.match(r"^\d+$", pos):
            continue
        rec = parse_result_row(pos, body, race_id)
        if not rec:
            continue
        a = cells[1].find("a", href=True)
        hid = None
        if a:
            hm = re.search(r"k_lineageLoginCode=(\w+)", a["href"])
            hid = hm.group(1) if hm else None

        entries.append({
            "race_id": race_id, "horse_no": rec["horse_no"],
            "bracket_no": rec["bracket_no"],
            "horse_id": hid, "horse_name": rec["horse_name"],
            "sex": None, "age": None,
            "weight_carried": rec["weight_carried"],
            "body_weight": None, "body_weight_dif": None,
            "jockey_id": None, "jockey_name": rec["jockey_name"],
            "trainer_id": None, "trainer_name": rec["trainer_name"],
            "belong_baba": None, "is_transfer": 0,
            "transfer_from": rec["belong"],
            "days_since_last": None, "last_baba_code": None,
            "last_finish_pos": None, "last_class_rank": None,
            "last_corner4": None,
        })
        results.append({
            "race_id": race_id, "horse_no": rec["horse_no"],
            "finish_pos": rec["finish_pos"], "dnf_reason": None,
            "time_sec": rec["time_sec"], "margin": rec["margin"],
            "last_3f": rec["last_3f"],
            "corner_1": None, "corner_2": None,
            "corner_3": None, "corner_4": None,
            "final_win_odds": None,      # オッズページで後から埋める
            "final_popularity": rec["final_popularity"],
        })

    if not entries:
        raise ParseError("成績行が0件", dump(html, f"norows-{race_id}"))

    race = {
        "race_id": race_id, "race_date": race_date.isoformat(),
        "baba_code": baba, "baba_name": NAR_BABA.get(baba, str(baba)),
        "race_no": race_no, "post_time": post_time, "race_name": race_name,
        "surface": surface, "distance_m": distance, "direction": direction,
        "course_note": None,
        "grade": None, "class_code": None, "class_rank": None,
        "age_cond": None, "sex_cond": None,
        "weather": wea.group(1) if wea else None,
        "track_cond": trk.group(1) if trk else None,
        "track_cond_num": {"良": 1, "稍重": 2, "重": 3, "不良": 4}.get(
            trk.group(1)) if trk else None,
        "n_starters": len(entries), "prize_1st": None,
        "fetched_at": dt.datetime.now().isoformat(),
    }
    return {"race": race, "entries": entries, "results": results}


# =====================================================================
# オッズページ
# =====================================================================
def parse_odds_page(html: str, race_id: str, bet: str,
                    captured_at: str, tag: str = "final") -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []

    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue

        if bet in ("win", "place"):
            if not re.match(r"^\d+$", cells[0]):
                continue
            combo = str(int(cells[0]))
        else:
            nums = re.findall(r"\d+", cells[0])
            if not nums or not re.search(r"[-－‐]", cells[0]):
                continue
            combo = "-".join(str(int(n)) for n in nums)

        vals = re.findall(r"\d+(?:\.\d+)?", cells[-1].replace(",", ""))
        if not vals:
            continue
        out.append({
            "race_id": race_id, "bet_type": bet, "combination": combo,
            "odds": float(vals[0]),
            "odds_upper": float(vals[1]) if len(vals) > 1 and bet == "place"
            else None,
            "snapshot_tag": tag, "captured_at": captured_at,
            "min_to_post": None,
        })
    return out


# =====================================================================
# 払戻ページ
# =====================================================================
BET_JP = {
    "単勝": "win", "複勝": "place",
    "枠連複": "bracket_quinella", "枠連単": "bracket_exacta",
    "馬連複": "quinella", "馬連単": "exacta",
    "馬単": "exacta", "馬連": "quinella",
    "ワイド": "wide", "三連複": "trio", "三連単": "trifecta",
}


def parse_payout_page(html: str, race_id: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 2:
            continue
        bet = BET_JP.get(cells[0].replace(" ", ""))
        if not bet:
            continue
        nums = re.findall(r"\d+", cells[1])
        yens = re.findall(r"[\d,]+", cells[2]) if len(cells) > 2 else []
        if not nums or not yens:
            continue
        out.append({
            "race_id": race_id, "bet_type": bet,
            "combination": "-".join(str(int(n)) for n in nums),
            "payout_yen": int(yens[0].replace(",", "")),
            "popularity": int(cells[3]) if len(cells) > 3
            and cells[3].isdigit() else None,
        })
    return out


# =====================================================================
# 日別レース一覧
# =====================================================================
RACE_NO_RE = re.compile(r"k_raceNo=(\d+)")


def parse_race_list(html: str) -> list[int]:
    """その日そこで行われたレース番号一覧。非開催なら空。"""
    nos = {int(n) for n in RACE_NO_RE.findall(html)}
    return sorted(n for n in nos if 1 <= n <= 12)
