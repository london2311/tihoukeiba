"""
取得層。

★重要な前提★
  このモジュールの HTTP 部・リトライ・レート制御・文字コード処理は検証済みだが、
  DOM セレクタは実サイトに当てて1回だけ校正する必要がある。
  そのため各パーサは「失敗したら HTML を dump して、何が見つからなかったか言う」
  自己診断型にしてある。初回は必ず calibrate.py を先に走らせること。

レート制御について:
  netkeiba も keiba.go.jp も個人運営ではない。2秒間隔を下回らないこと。
  10000レースなら約6時間。急ぐ理由は無い。ブロックされる方が遥かに高くつく。
"""
from __future__ import annotations

import re
import time
import random
import datetime as dt
from pathlib import Path
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

from .core import JST, BABA, CLASS_RANK, TRACK_COND_NUM, make_race_id

DUMP_DIR = Path(__file__).resolve().parent.parent / "data" / "dumps"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"


class ParseError(Exception):
    """パース失敗。dump_path に当該HTMLを保存してある。"""
    def __init__(self, msg: str, dump_path: Path | None = None):
        super().__init__(f"{msg}" + (f"  [dump: {dump_path}]" if dump_path else ""))
        self.dump_path = dump_path


# =====================================================================
# HTTP
# =====================================================================
class Fetcher:
    def __init__(self, min_interval: float = 2.0, jitter: float = 0.8,
                 timeout: float = 20.0, max_retry: int = 4):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA,
                               "Accept-Language": "ja,en;q=0.8"})
        self.min_interval = min_interval
        self.jitter = jitter
        self.timeout = timeout
        self.max_retry = max_retry
        self._last = 0.0

    def _wait(self):
        elapsed = time.monotonic() - self._last
        sleep = self.min_interval - elapsed + random.uniform(0, self.jitter)
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.monotonic()

    def get(self, url: str, encoding: str | None = None) -> str:
        last_exc = None
        for attempt in range(self.max_retry):
            self._wait()
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 404:
                    raise ParseError(f"404 {url}")
                if r.status_code in (429, 503):
                    # 明示的に絞られている。素直に長く待つ
                    back = 30 * (2 ** attempt)
                    print(f"    [{r.status_code}] backing off {back}s")
                    time.sleep(back)
                    continue
                r.raise_for_status()
                # netkeiba は EUC-JP、keiba.go.jp は UTF-8
                r.encoding = encoding or r.apparent_encoding
                return r.text
            except requests.RequestException as e:
                last_exc = e
                time.sleep(3 * (2 ** attempt))
        raise ParseError(f"fetch failed after {self.max_retry}: {url} ({last_exc})")


def dump(html: str, tag: str) -> Path:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    p = DUMP_DIR / f"{tag}-{dt.datetime.now(JST):%Y%m%d%H%M%S}.html"
    p.write_text(html, encoding="utf-8")
    return p


# =====================================================================
# 正規化ユーティリティ (ここは実データ由来のゆらぎを吸収する層。重要)
# =====================================================================
def _num(s, cast=float):
    """'1.5' '1,234' '54.0kg' '--' '' -> 値 or None"""
    if s is None:
        return None
    t = re.sub(r"[^\d.\-]", "", str(s).replace(",", ""))
    if t in ("", "-", ".", "-."):
        return None
    try:
        return cast(float(t))
    except ValueError:
        return None


def parse_time_sec(s: str | None) -> float | None:
    """'1:52.3' や '52.3' -> 秒"""
    if not s:
        return None
    m = re.match(r"(?:(\d+):)?(\d+)\.(\d)", s.strip())
    if not m:
        return None
    mi, sec, dec = m.group(1), m.group(2), m.group(3)
    return (int(mi) * 60 if mi else 0) + int(sec) + int(dec) / 10


def parse_body_weight(s: str | None) -> tuple[int | None, int | None]:
    """'488(+4)' -> (488, 4)"""
    if not s:
        return None, None
    m = re.match(r"\s*(\d+)\s*\(?\s*([+-]?\d+)?\s*\)?", s)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def parse_sex_age(s: str | None) -> tuple[str | None, int | None]:
    """'牡3' 'セ5' -> ('牡', 3)"""
    if not s:
        return None, None
    m = re.match(r"\s*([牡牝セせ])\s*(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def parse_corners(s: str | None) -> list[int | None]:
    """'3-3-2-1' -> [3,3,2,1] / 足りない分は None 詰め"""
    if not s:
        return [None] * 4
    parts = [int(x) for x in re.findall(r"\d+", s)][:4]
    return parts + [None] * (4 - len(parts))


def normalize_class(text: str) -> tuple[str | None, str | None, int | None]:
    """レース条件文字列 -> (grade, class_code, class_rank)"""
    if not text:
        return None, None, None
    grade = None
    for g in ("JpnI", "JpnII", "JpnIII", "GI", "GII", "GIII"):
        if g in text:
            grade = g.replace("G", "Jpn") if g.startswith("G") else g
            break
    code = None
    m = re.search(r"\b([ABCD])\s*([123])\b", text)
    if m:
        code = f"{m.group(1)}{m.group(2)}"
    elif grade:
        code = grade
    return grade, code, CLASS_RANK.get(code or grade or "", None)


def parse_course(text: str) -> dict:
    """'ダ左1600m' / '芝右1400m' -> surface/direction/distance"""
    out = {"surface": None, "direction": None, "distance_m": None}
    if not text:
        return out
    if "ダ" in text:
        out["surface"] = "dirt"
    elif "芝" in text:
        out["surface"] = "turf"
    if "左" in text:
        out["direction"] = "left"
    elif "右" in text:
        out["direction"] = "right"
    elif "直" in text:
        out["direction"] = "straight"
    m = re.search(r"(\d{3,4})\s*m", text)
    if m:
        out["distance_m"] = int(m.group(1))
    return out


# =====================================================================
# パーサ本体
#   セレクタは SEL 辞書に集約。校正時はここだけ直せば済むようにしてある。
# =====================================================================
SEL = {
    # --- netkeiba 地方 (db.netkeiba.com/race/<id>/) ---
    "nk_race_title":  "diary_snap_cut, .racedata h1, .data_intro h1",
    "nk_race_cond":   ".racedata diary_snap_cut span, .data_intro p span",
    "nk_result_tbl":  "table.race_table_01, table.RaceTable01",
    "nk_payout_tbl":  "table.pay_table_01, .Payout_Detail_Table",
    # --- keiba.go.jp (NAR公式) ---
    "nar_result_tbl": "table.tb01, table.raceTable",
}

# 結果テーブルの列名 -> 内部キー。表記ゆれを全部ここで吸収する
COLMAP = {
    "着順": "finish_pos", "着": "finish_pos",
    "枠番": "bracket_no", "枠": "bracket_no",
    "馬番": "horse_no", "馬": "horse_no",
    "馬名": "horse_name",
    "性齢": "sex_age",
    "斤量": "weight_carried", "負担重量": "weight_carried",
    "騎手": "jockey_name",
    "タイム": "time_str", "走破時計": "time_str",
    "着差": "margin",
    "通過": "corner_str", "通過順位": "corner_str", "コーナー通過順位": "corner_str",
    "上り": "last_3f", "上がり": "last_3f", "後3F": "last_3f",
    "単勝": "final_win_odds", "単勝オッズ": "final_win_odds", "オッズ": "final_win_odds",
    "人気": "final_popularity",
    "馬体重": "body_weight_str", "体重": "body_weight_str",
    "調教師": "trainer_name", "厩舎": "trainer_name",
    # --- 実データで確認された別名 ---
    "推定上り": "last_3f", "上り3F": "last_3f", "上がり3F": "last_3f",
    "確定着順": "finish_pos", "着順同着": "finish_pos",
    "単勝人気": "final_popularity", "人気順": "final_popularity",
    "所属": "belong_baba", "調教場所": "belong_baba",
}


def norm_header(h: str) -> str:
    """ヘッダ名の正規化。全角/半角空白、改行、括弧書きを落とす。
    実データは '着 順' '馬 番' '上り(3F)' のような揺れが普通にある。"""
    h = re.sub(r"[\s\u3000]+", "", h)
    h = re.sub(r"[（(].*?[)）]", "", h)
    return h


def _table_to_dicts(table) -> list[dict]:
    """<table> -> ヘッダ名でキー付けした行辞書のリスト。列順の変化に強い。"""
    rows = table.find_all("tr")
    if not rows:
        return []
    header_cells = rows[0].find_all(["th", "td"])
    headers = [norm_header(c.get_text(strip=True)) for c in header_cells]
    keys = [COLMAP.get(h, f"_{i}_{h}") for i, h in enumerate(headers)]

    out = []
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        rec = {}
        for k, c in zip(keys, cells):
            rec[k] = c.get_text(" ", strip=True)
            a = c.find("a", href=True)
            if a:
                rec[k + "_href"] = a["href"]
        out.append(rec)
    return out


def _extract_id(href: str | None, kind: str) -> str | None:
    """/horse/2021104567/ -> '2021104567'"""
    if not href:
        return None
    m = re.search(rf"/{kind}/(?:result/)?([0-9a-zA-Z]+)", href)
    return m.group(1) if m else None


def parse_race_page(html: str, race_date: dt.date, baba_code: int,
                    race_no: int, source_url: str = "") -> dict:
    """
    レース結果ページ -> {races, entries, results, payouts}
    確定単勝オッズと人気は全馬ぶんここで取れる。これが今回の主データ。
    """
    soup = BeautifulSoup(html, "html.parser")
    race_id = make_race_id(race_date, baba_code, race_no)

    # --- 結果テーブル ---
    table = None
    for sel in SEL["nk_result_tbl"].split(", "):
        table = soup.select_one(sel)
        if table:
            break
    if table is None:
        raise ParseError("結果テーブルが見つからない。SEL['nk_result_tbl'] を校正せよ",
                         dump(html, f"noresulttbl-{race_id}"))

    rows = _table_to_dicts(table)
    if not rows:
        raise ParseError("結果テーブルが空", dump(html, f"emptytbl-{race_id}"))

    missing = {"finish_pos", "horse_no", "horse_name"} - set(rows[0].keys())
    if missing:
        raise ParseError(f"必須列が取れない: {missing} / 実ヘッダ={list(rows[0])}",
                         dump(html, f"colmap-{race_id}"))

    # --- レース条件 ---
    head_text = " ".join(
        e.get_text(" ", strip=True)
        for e in soup.select("h1, .data_intro, .racedata, .RaceData01, .RaceData02")
    )
    course = parse_course(head_text)
    grade, class_code, class_rank = normalize_class(head_text)
    cond_m = re.search(r"(良|稍重|重|不良)", head_text)
    weather_m = re.search(r"天候\s*:?\s*(\S+)", head_text)
    post_m = re.search(r"(\d{1,2}):(\d{2})", head_text)
    title_el = soup.select_one("h1")

    entries, results = [], []
    for r in rows:
        hno = _num(r.get("horse_no"), int)
        if hno is None:
            continue
        sex, age = parse_sex_age(r.get("sex_age"))
        bw, bwd = parse_body_weight(r.get("body_weight_str"))
        c1, c2, c3, c4 = parse_corners(r.get("corner_str"))
        fin = _num(r.get("finish_pos"), int)

        entries.append({
            "race_id": race_id, "horse_no": hno,
            "bracket_no": _num(r.get("bracket_no"), int),
            "horse_id": _extract_id(r.get("horse_name_href"), "horse"),
            "horse_name": r.get("horse_name"),
            "sex": sex, "age": age,
            "weight_carried": _num(r.get("weight_carried")),
            "body_weight": bw, "body_weight_dif": bwd,
            "jockey_id": _extract_id(r.get("jockey_name_href"), "jockey"),
            "jockey_name": r.get("jockey_name"),
            "trainer_id": _extract_id(r.get("trainer_name_href"), "trainer"),
            "trainer_name": r.get("trainer_name"),
            "belong_baba": None, "is_transfer": 0, "transfer_from": None,
            "days_since_last": None, "last_baba_code": None,
            "last_finish_pos": None, "last_class_rank": None, "last_corner4": None,
        })
        results.append({
            "race_id": race_id, "horse_no": hno,
            "finish_pos": fin,
            "dnf_reason": None if fin else (r.get("finish_pos") or None),
            "time_sec": parse_time_sec(r.get("time_str")),
            "margin": r.get("margin"),
            "last_3f": _num(r.get("last_3f")),
            "corner_1": c1, "corner_2": c2, "corner_3": c3, "corner_4": c4,
            "final_win_odds": _num(r.get("final_win_odds")),
            "final_popularity": _num(r.get("final_popularity"), int),
        })

    race = {
        "race_id": race_id,
        "race_date": race_date.isoformat(),
        "baba_code": baba_code,
        "baba_name": BABA.get(baba_code, str(baba_code)),
        "race_no": race_no,
        "post_time": f"{post_m.group(1)}:{post_m.group(2)}" if post_m else None,
        "race_name": title_el.get_text(strip=True) if title_el else None,
        "surface": course["surface"], "distance_m": course["distance_m"],
        "direction": course["direction"], "course_note": None,
        "grade": grade, "class_code": class_code, "class_rank": class_rank,
        "age_cond": None, "sex_cond": None,
        "weather": weather_m.group(1) if weather_m else None,
        "track_cond": cond_m.group(1) if cond_m else None,
        "track_cond_num": TRACK_COND_NUM.get(cond_m.group(1)) if cond_m else None,
        "n_starters": len(entries),
        "prize_1st": None,
        "fetched_at": dt.datetime.now(JST).isoformat(),
    }

    if race["distance_m"] is None:
        raise ParseError("距離が取れない。条件テキストのセレクタを校正せよ",
                         dump(html, f"nodist-{race_id}"))

    return {"race": race, "entries": entries, "results": results,
            "payouts": parse_payouts(soup, race_id)}


BET_JP = {
    "単勝": "win", "複勝": "place", "馬連": "quinella", "馬単": "exacta",
    "ワイド": "wide", "枠連": "bracket_quinella",
    "三連複": "trio", "3連複": "trio",
    "三連単": "trifecta", "3連単": "trifecta",
}


def parse_payouts(soup, race_id: str) -> list[dict]:
    """払戻テーブル。1レースあたり各券種の的中目のオッズが1点ずつ得られる。
    三連系オッズの再構成モデルを較正する唯一の実測点になる。"""
    out = []
    tables = []
    for sel in SEL["nk_payout_tbl"].split(", "):
        tables += soup.select(sel)
    for tbl in tables:
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            label = cells[0].get_text(strip=True)
            bet = BET_JP.get(label)
            if not bet:
                continue
            combos = [x for x in cells[1].get_text("\n", strip=True).split("\n") if x]
            yens = re.findall(r"[\d,]+", cells[2].get_text("\n", strip=True)) \
                if len(cells) > 2 else []
            pops = re.findall(r"\d+", cells[3].get_text("\n", strip=True)) \
                if len(cells) > 3 else []
            for i, cmb in enumerate(combos):
                nums = re.findall(r"\d+", cmb)
                if not nums or i >= len(yens):
                    continue
                out.append({
                    "race_id": race_id, "bet_type": bet,
                    "combination": "-".join(str(int(n)) for n in nums),
                    "payout_yen": int(yens[i].replace(",", "")),
                    "popularity": int(pops[i]) if i < len(pops) else None,
                })
    return out
