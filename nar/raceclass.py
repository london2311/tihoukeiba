"""
race_name からクラス条件を復元する。

★背景★
  keiba.go.jp の成績ページは「園田競馬場 2026年8月13日」の次の行に
  条件行を置いている。fetchers.py はそれを race_name として保存している。
  実物:
      'Ｃ３一３歳以上'          園田
      'いわて北緯４０度葛巻町賞Ｂ２'  盛岡
      'Ｃ３－９'                高知
      'Ｂ８組'                  名古屋   ← ★体系が違う★
  つまりクラスは捨てられていない。文字列から取り出せる。

★場によって体系が違う（ここを間違えると class_move が嘘になる）★
  多数派(盛岡/水沢/浦和/船橋/大井/川崎/金沢/園田/姫路/高知/佐賀):
      英字+数字 = クラス。 'Ｃ２一' は C2クラスの一組
  名古屋(24)/笠松(23):
      英字1文字 = クラス、数字 = 組番号。 'Ｃ１５組' は Cクラスの15組
      → ここで 'Ｃ１組' を C1クラスと読むと誤り。
        欠損より有害なので、この2場は英字1文字だけを見る。

診断ログ(2026-08-15, 16,467レース)での実測:
  英字+数字で拾えた 64.1%。うち名古屋11.6%/笠松17.0%は上記の誤読。
  この分岐を入れると全体 85%前後まで上がる。
"""
from __future__ import annotations

import re
import unicodedata

# 組番号を「クラスの数字」と誤読してはいけない場
GROUP_STYLE_BABA = {23, 24}          # 笠松, 名古屋

# 小さいほど強い。within-race で見るのは class_move なので
# 絶対値そのものより「差」が意味を持つ
CLASS_RANK: dict[str, int] = {
    "JpnI": 0, "JpnII": 1, "JpnIII": 2, "重賞": 3,
    "OPEN": 5,
    "A1": 10, "A2": 11, "A3": 12, "A": 11,
    "B1": 20, "B2": 21, "B3": 22, "B": 21,
    "C1": 30, "C2": 31, "C3": 32, "C": 31,
    "D1": 40, "D2": 41, "D3": 42, "D": 41,
    # 年齢条件戦。強さの梯子とは別物なので is_age_race で区別できるようにする
    "3歳": 34, "2歳": 44, "未勝利": 48, "新馬": 52,
}
AGE_CODES = {"3歳", "2歳", "未勝利", "新馬"}

_AB_NUM = re.compile(r"(?<![A-Za-z])([ABCD])\s?([1-3])(?![0-9])")
_AB_ONLY = re.compile(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])")

_GRADES = [
    ("JpnIII", re.compile(r"JpnIII|GIII|G3")),
    ("JpnII",  re.compile(r"JpnII(?!I)|GII(?!I)|G2")),
    ("JpnI",   re.compile(r"JpnI(?!I)|GI(?!I)|G1")),
    ("重賞",   re.compile(r"重賞")),
]


def classify(race_name: str | None, baba_code: int | None = None
             ) -> tuple[str | None, str]:
    """race_name -> (class_code, 適用ルール)

    class_code は CLASS_RANK のキー。取れなければ (None, 'unmatched')。
    baba_code を渡すと名古屋/笠松の組番号誤読を避ける。
    """
    if not race_name:
        return None, "empty"
    s = unicodedata.normalize("NFKC", str(race_name))

    for code, rx in _GRADES:
        if rx.search(s):
            return code, "grade"

    if baba_code in GROUP_STYLE_BABA:
        # 数字は組番号。英字1文字だけを信用する
        m = _AB_ONLY.search(s)
        if m:
            return m.group(1), "letter_only"
    else:
        m = _AB_NUM.search(s)
        if m:
            return f"{m.group(1)}{m.group(2)}", "letter_num"
        # 'Ｂ級サバイバル選抜' のような数字なしの表記を拾う
        m = _AB_ONLY.search(s)
        if m and ("級" in s or "組" in s):
            return m.group(1), "letter_only"

    if re.search(r"オープン|OP(?![A-Za-z])", s):
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


def rank_of(race_name: str | None, baba_code: int | None = None) -> int | None:
    code, _ = classify(race_name, baba_code)
    return CLASS_RANK.get(code) if code else None


def is_age_race(race_name: str | None, baba_code: int | None = None) -> int:
    code, _ = classify(race_name, baba_code)
    return int(code in AGE_CODES)
