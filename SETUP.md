# セットアップ手順

## 1. push

```bash
git clone https://github.com/<あなた>/<リポジトリ名>.git
cd <リポジトリ名>
# ダウンロードした nar-collector の中身をここに展開
git add -A
git commit -m "init: NAR collector"
git push
```

**リポジトリは Public にすること。**
Private の Actions 無料枠は月2,000分。6h × 6ジョブ = 2,160分で初日に枯れる。

## 2. セレクタ校正（これだけは手作業。10分）

ローカルで1レースだけ実行する。

```bash
pip install -r requirements.txt
python -m nar.calibrate --date 2026-08-01 --baba 44 --race 11
```

出力の見方:

- `■ COLMAP に未登録のヘッダ` が出たら → `nar/fetchers.py` の `COLMAP` に追記
- `■ SEL['nk_result_tbl'] に ... を追加` が出たら → `SEL` に追記
- `!! <a href> が無い` が出たら → **重大**。horse_id が取れず過去走特徴が作れない
- 最後に `✓ 成功` + `校正完了` が出れば終わり

**2〜3場でやること。** 大井(44)・園田(50)・高知(54) あたりは
テーブル構造が微妙に違う。1場で通っても他場で落ちることがある。

```bash
python -m nar.calibrate --date 2026-08-01 --baba 50 --race 11
python -m nar.calibrate --date 2026-08-01 --baba 54 --race 11
```

直したら commit & push。

## 3. 初回実行

GitHub → **Actions** タブ → `NAR backfill (sharded)` → **Run workflow**

| 入力 | 値 |
|---|---|
| since | `2023-01-01` |
| until | `2026-08-01` |
| limit | `12000` |
| shards | `6` |

流れ: `seed`(開催日先読み) → `collect`(6並列 各5.5h) → `merge`(統合+リーク検査)

一晩で約6万レース。3年分なら3晩。

## 4. 2日目以降

何もしなくていい。毎日 02:00 JST に自動で続きが走る。
`since` を空にしておけば既存キューの残りを消化する。

進捗確認:

```bash
# Actions の merge ジョブのログに coverage が出る
# ローカルで見るなら Artifact 'nar-db-main' を落として
python -m nar.backfill --report
```

## 詰まりやすい所

| 症状 | 原因 | 対処 |
|---|---|---|
| collect が全件 skipped | seed が動いていない | `since` を入れて再実行 |
| 429/503 が頻発 | 並列度が高い | shards を 3 に下げる |
| merge が失敗し始める | DBが Artifact 上限に近い | Releases か Parquet に移行 |
| parse_error が大量 | 校正漏れの場がある | `data/dumps/` の HTML を calibrate に食わせる |
