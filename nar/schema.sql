-- =====================================================================
-- 地方競馬(NAR) データ収集スキーマ
--
-- 設計方針:
--   races / entries / results  … 過去バックフィル可能(オッズ不要)
--   odds_snapshots             … 前方収集のみ。取得時刻が生命線
--
-- race_id 形式: YYYYMMDD-BB-RR   (BB=場コード, RR=レース番号 2桁)
--   例) 20260814-44-11  = 2026/08/14 大井 11R
-- =====================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- レース基本情報
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    race_date       TEXT NOT NULL,          -- YYYY-MM-DD
    baba_code       INTEGER NOT NULL,       -- 30=門別 44=大井 50=園田 ...
    baba_name       TEXT NOT NULL,
    race_no         INTEGER NOT NULL,
    post_time       TEXT,                   -- HH:MM 発走時刻
    race_name       TEXT,

    -- コース条件(バイアス分析の主軸)
    surface         TEXT,                   -- dirt / turf
    distance_m      INTEGER NOT NULL,
    direction       TEXT,                   -- right / left / straight
    course_note     TEXT,                   -- 内回り/外回り 等

    -- クラス条件
    grade           TEXT,                   -- JpnI/JpnII/JpnIII/重賞/NULL
    class_code      TEXT,                   -- A1 B2 C3 / 2歳 / 一般 など生値
    class_rank      INTEGER,                -- 正規化した強さ順位(自前採番)
    age_cond        TEXT,
    sex_cond        TEXT,

    -- 環境
    weather         TEXT,
    track_cond      TEXT,                   -- 良/稍重/重/不良
    track_cond_num  INTEGER,                -- 1..4 数値化

    n_starters      INTEGER NOT NULL,
    prize_1st       INTEGER,

    fetched_at      TEXT NOT NULL,
    UNIQUE (race_date, baba_code, race_no)
);
CREATE INDEX IF NOT EXISTS idx_races_date  ON races(race_date);
CREATE INDEX IF NOT EXISTS idx_races_baba  ON races(baba_code, distance_m);

-- ---------------------------------------------------------------------
-- 出走馬(レース前に確定する情報のみ。ここに結果を混ぜない)
--   ※リーク防止のため results と物理的にテーブルを分ける
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entries (
    race_id         TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    horse_no        INTEGER NOT NULL,       -- 馬番
    bracket_no      INTEGER,                -- 枠番
    horse_id        TEXT,                   -- 血統登録番号(名寄せの主キー)
    horse_name      TEXT NOT NULL,

    sex             TEXT,
    age             INTEGER,
    weight_carried  REAL,                   -- 斤量
    body_weight     INTEGER,                -- 馬体重
    body_weight_dif INTEGER,                -- 増減

    jockey_id       TEXT,
    jockey_name     TEXT,
    trainer_id      TEXT,
    trainer_name    TEXT,
    belong_baba     INTEGER,                -- 所属場コード
    is_transfer     INTEGER DEFAULT 0,      -- 転入初戦フラグ
    transfer_from   TEXT,                   -- 'JRA' / 他地区場コード

    -- 前走情報(市場の遅行評価を突く材料)
    days_since_last INTEGER,
    last_baba_code  INTEGER,
    last_finish_pos INTEGER,
    last_class_rank INTEGER,
    last_corner4    INTEGER,                -- 前走4角通過順(脚質推定の素)

    PRIMARY KEY (race_id, horse_no)
);
CREATE INDEX IF NOT EXISTS idx_entries_horse  ON entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_entries_jockey ON entries(jockey_id);

-- ---------------------------------------------------------------------
-- 結果
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results (
    race_id         TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    horse_no        INTEGER NOT NULL,

    finish_pos      INTEGER,                -- NULL = 中止/除外
    dnf_reason      TEXT,
    time_sec        REAL,
    margin          TEXT,                   -- 着差
    last_3f         REAL,                   -- 上がり3F
    corner_1        INTEGER,
    corner_2        INTEGER,
    corner_3        INTEGER,
    corner_4        INTEGER,

    -- 確定オッズ(スナップショットの取りこぼし保険。EV検証の主データにはしない)
    final_win_odds  REAL,
    final_popularity INTEGER,

    PRIMARY KEY (race_id, horse_no)
);

-- ---------------------------------------------------------------------
-- 払戻(モデル検証の答え合わせ用)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS payouts (
    race_id         TEXT NOT NULL REFERENCES races(race_id) ON DELETE CASCADE,
    bet_type        TEXT NOT NULL,          -- win/place/quinella/exacta/wide/trio/trifecta
    combination     TEXT NOT NULL,          -- '3' / '3-7' / '3-7-11' (馬番を'-'連結)
    payout_yen      INTEGER NOT NULL,       -- 100円あたり
    popularity      INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

-- ---------------------------------------------------------------------
-- オッズスナップショット ★このテーブルが本丸★
--
--   同一レースを複数時点で取る。
--   snapshot_tag: 'morning'(発売直後) / 'pre30' / 'pre02'(締切2分前)
--   min_to_post : 発走までの残り分数。ドリフト特徴はこれで正規化する
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
    race_id         TEXT NOT NULL,
    bet_type        TEXT NOT NULL,
    combination     TEXT NOT NULL,
    odds            REAL,                   -- 複勝は下限値
    odds_upper      REAL,                   -- 複勝の上限値。他はNULL
    snapshot_tag    TEXT NOT NULL,
    captured_at     TEXT NOT NULL,          -- ISO8601 (JST)
    min_to_post     INTEGER,
    PRIMARY KEY (race_id, bet_type, combination, snapshot_tag)
);
CREATE INDEX IF NOT EXISTS idx_odds_race ON odds_snapshots(race_id, bet_type);

-- ---------------------------------------------------------------------
-- 収集ログ(欠損の自己申告。これが無いと後で欠損とゼロの区別がつかなくなる)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fetch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target          TEXT NOT NULL,          -- race_id or date
    kind            TEXT NOT NULL,          -- entries/results/odds/payouts
    status          TEXT NOT NULL,          -- ok/empty/http_error/parse_error
    detail          TEXT,
    logged_at       TEXT NOT NULL
);

-- ---------------------------------------------------------------------
-- バックフィル進捗 (再開可能性の担保)
--   10000レース収集は数時間かかる。必ず途中で落ちる前提で設計する。
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS checkpoints (
    scope           TEXT PRIMARY KEY,       -- 'calendar' / 'race'
    key             TEXT NOT NULL,          -- 'YYYY-MM' or race_id
    status          TEXT NOT NULL,          -- pending/done/failed/skipped
    attempts        INTEGER DEFAULT 0,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS race_queue (
    race_id         TEXT PRIMARY KEY,
    race_date       TEXT NOT NULL,
    baba_code       INTEGER NOT NULL,
    race_no         INTEGER NOT NULL,
    source_url      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempts        INTEGER DEFAULT 0,
    last_error      TEXT,
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON race_queue(status, race_date);
