# 研究ノート

## 2026-10-12：フェーズ1・フェーズ2

### やったこと

- フェーズ0の再確認：プロジェクトフォルダに venv が存在しなかったため、Python 3.12 で venv を作り直し、
  `minigrid` 3.1.0 / `gymnasium` 1.4.0 / `stable-baselines3` 2.9.0 をインストールした。
- フェーズ1：`MiniGrid-DoorKey-6x6-v0` の仕様をソース（`minigrid/envs/doorkey.py`,
  `minigrid/minigrid_env.py`, `minigrid/wrappers.py`）から確認した。
- フェーズ2：設計判断（観測空間、ログに画像を含めるか）をユーザーに確認の上、
  `recorder.py`（`EpisodeRecorder` という Gymnasium Wrapper）を実装し、ランダム方策で
  複数エピソード記録して項目を検証した。

### 環境メモ（詰まった点）

- このマシンでは Windows 11 の **Smart App Control**（レジストリ
  `HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState=1`）が
  有効になっており、pipで入れた最新の torch（2.14.1）のDLL（`torch_python.dll` 等）が
  署名なし（`Get-AuthenticodeSignature` で `NotSigned`）のためロードをブロックされた
  （`OSError: [WinError 4551] ...ポリシーによってこのファイルはブロックされました`）。
  `torch==2.8.0` に固定したところ読み込めるようになった。おそらく評価版/最新版よりも
  枯れたバージョンの方がMicrosoftの評判情報に載っているためと思われる。
  → `requirements` を書くときは `torch==2.8.0`（CPU版）で固定しておくこと。
  Smart App Controlを無効化する手は検討していない（一度無効化するとWindowsの再インストールが
  必要になる一方通行の設定のため、安易に触らない方がいい）。

### フェーズ1の確認結果（仕様まとめ）

- 観測空間（デフォルト・部分観測）：`Dict(image: Box(0,255,(7,7,3),uint8), direction: Discrete(4), mission: str)`
  各セルは `(OBJECT_IDX, COLOR_IDX, STATE)`。`FullyObsWrapper` を付けると `image` が
  `(grid_width, grid_height, 3)` の全体マップになる（6x6なら `(6,6,3)`、実機で形状確認済み）。
- 行動空間：`Discrete(7)` = `left(0) right(1) forward(2) pickup(3) drop(4) toggle(5) done(6)`。
  DoorKeyでは `drop` と `done` は未使用（docstringに明記）。
- 報酬：ゴール到達時のみ `1 - 0.9*(step_count/max_steps)`。それ以外は `0`（スパース）。
- 終了条件：`terminated` はゴール到達時のみ `True`。`truncated` は `step_count >= max_steps`
  （6x6では `max_steps=360` = `10*size**2`）。ランダム方策で実機確認し、360stepでtruncatedになった。
- Wrapper確認：
  - `FullyObsWrapper`：部分観測(7x7)を全体観測（グリッドサイズ）に変換。実機で形状確認済み。
  - `SymbolicObsWrapper`：全体観測だが `(x, y, object_idx)` のみで色やドア状態を含まない。今回は不採用。
  - `ReseedWrapper`：`reset()` を引数なしで呼んでも固定seedリストを強制するラッパー。
    検証した結果、`env.reset(seed=N)` を毎回明示的に渡すだけで同一レイアウトが再現されることを
    確認した（グリッドのエンコードが完全一致）。今回はエピソードごとにseedを明示的に渡すので
    `ReseedWrapper` は使わない。

### ユーザー確認した設計判断

1. **観測空間**：全体観測（`FullyObsWrapper`）を採用。人間はGUIで画面全体を見ながらプレイするのが
   自然なので、AIにも同じ情報量を与えて比較を公平にする。
   （MiniGridの標準ベンチマーク（部分観測が前提）とは条件が変わるが、今回の目的は人間との比較であり
   既存研究の数値との比較ではないので問題ない）
2. **ログの内容**：構造化データのみ。生のグリッド画像配列はログに保存しない
   （seedと行動列から後で再現できるため。ログを軽く・読みやすく保つ）。

### recorder.py の実装と確認結果

- `EpisodeRecorder`（Gymnasium Wrapper）を実装。`reset()` で新しいエピソードの開始行（`step=0`、
  `action=None`）を書き、`step()` で各stepの結果行を書く。1エピソード=複数行、1行=1step。
- ログの値は常に `env.unwrapped` のgrid/agent状態から直接取得するようにした。これにより
  「ポリシーに見せる観測（部分/全体）」と「ログに残す情報」を分離している
  （観測空間をどちらに選んでも、ログの内容は変わらない）。
- 検証：`MiniGrid-DoorKey-6x6-v0` + `FullyObsWrapper` + `EpisodeRecorder` でランダム方策を
  20エピソード実行し、保存されたJSONLを読み返して確認した。
  - `episode` がエピソードごとに正しく1,2,3,...と増え、各エピソード内で `seed` が一定であることを確認。
  - `pickup` で `carrying: null -> "key"`、`drop` で `"key" -> null` に正しく変化することを確認。
  - `toggle` で `door_state: "locked" -> "open"`、鍵を持たず再度toggleすると `"open" -> "closed"` に
    正しく変化することを確認（`Door.toggle()` の仕様どおり）。
  - 20エピソード中、ランダム方策でゴールに到達した（`terminated=True`）エピソードは0件。
    DoorKeyはスパース報酬で難しいというCLAUDE.mdの記述と一致する結果。

### 未確定・次回への引き継ぎ

- Gemini APIキーの取得・動作確認はまだ行っていない（フェーズ5で使う）。
- `.env` ファイルは未作成。`GEMINI_API_KEY` を読む実装はまだない。
- `train_agent.py` / `run_agent.py` / `play_human.py` / `analyze_stats.py` / `ask_llm.py` は未実装。
- `requirements.txt`（`pip freeze`、`torch==2.8.0` 固定を含む）は書き出し済み。

### 次にやること（フェーズ3の提案）

- `train_agent.py`：PPO（`FullyObsWrapper` + `ImgObsWrapper` or `FlatObsWrapper` 相当の前処理）で
  `MiniGrid-DoorKey-6x6-v0` を学習する。序盤・中盤・収束後の3チェックポイントでモデルを保存。
- `run_agent.py`：各チェックポイントのモデルで、固定シードリスト（例：seed 0〜49）を使って
  `EpisodeRecorder` 経由でプレイログを記録する。
- 学習前に、学習ステップ数・エピソード数・推定実行時間を見積もってから実行する
  （CLAUDE.mdの「まず小さく動かす」方針に従う）。
