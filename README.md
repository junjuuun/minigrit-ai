# minigrid-ai：できることまとめ（2026-10-12 時点）

このプロジェクトは「人間らしい振る舞いをするゲームAI」を作るための研究コード。
最終的には強化学習の報酬にLLMの「人間らしさスコア」を加えることを目指すが、
今はその準備段階（人間とAIのプレイログを記録する部分）まで進んでいる。

全体の計画・ルールは `claude.md` を参照。作業の経緯は `notes.md` を参照。
ここでは「今の時点で実際に動くもの」だけを分かりやすくまとめる。

## 今できること

### 1. MiniGridのDoorKey環境を動かす

`MiniGrid-DoorKey-6x6-v0` という環境が動作確認済み。

- 6x6マスの迷路に鍵・ドア・ゴールが配置される
- エージェントは「左を向く／右を向く／前に進む／拾う／置く／ドアを開閉する」の7種類の行動ができる
- 鍵を拾ってドアを開け、ゴールに着くと成功（報酬がもらえる）。それ以外は報酬0
- 同じ`seed`（乱数の種）を指定すれば、毎回同じ迷路レイアウトを再現できる
  → これがあるから「人間とAIに全く同じ迷路で遊んでもらって比較する」が可能になる

### 2. プレイの様子を記録する（`recorder.py`）

`recorder.py`に`EpisodeRecorder`という部品（Gymnasium Wrapper）がある。
これをMiniGridの環境にかぶせるだけで、プレイの一手一手がファイルに自動で記録される。

```python
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from recorder import EpisodeRecorder

env = gym.make("MiniGrid-DoorKey-6x6-v0")
env = FullyObsWrapper(env)                              # 画面全体が見える観測にする
env = EpisodeRecorder(env, "logs/agent/run.jsonl", source="agent")  # ここで記録が始まる

obs, info = env.reset(seed=0)        # 1エピソード開始（これだけで1行記録される）
action = env.action_space.sample()   # 行動を決める（本当はここでAIや人間が選ぶ）
obs, reward, terminated, truncated, info = env.step(action)  # 1歩進める（1行記録される）
env.close()
```

記録されるのは`logs/agent/run.jsonl`のようなテキストファイルで、1行が1歩（1step）に対応する。
中身はこんな感じ（実際に動作確認したログの例）：

```json
{"source": "agent", "episode": 1, "seed": 0, "step": 16, "pos_x": 1, "pos_y": 3,
 "direction": 0, "action": 3, "action_name": "pickup", "carrying": "key",
 "carrying_color": "yellow", "door_x": 3, "door_y": 1, "door_state": "locked",
 "reward": 0.0, "terminated": false, "truncated": false, "done": false, "t": 1791793233.1}
```

記録される項目：

| 項目 | 意味 |
|---|---|
| `source` | `"agent"`（AI）か`"human"`（人間）か |
| `episode` / `seed` | 何エピソード目か、迷路を再現するための種 |
| `step` | エピソード内の何手目か（0=開始直後、行動なし） |
| `pos_x`, `pos_y`, `direction` | エージェントの位置と向き |
| `action`, `action_name` | とった行動（番号と名前） |
| `carrying`, `carrying_color` | 今持っているもの（鍵など）。何も持っていなければ`null` |
| `door_x`, `door_y`, `door_state` | ドアの位置と状態（`locked`/`closed`/`open`） |
| `reward`, `terminated`, `truncated`, `done` | その手でもらった報酬と、エピソードが終わったか |
| `t` | 記録した時刻（AIの時間は人間の反応時間と比較できないので今は使わない） |

ポイント：**AIが見ている情報（部分観測か全体観測か）とログに残す情報は別物**にしている。
`FullyObsWrapper`を外しても、ログの内容（位置・持ち物・ドア状態など）は変わらない。
これは「ゲーム環境の内部状態」を直接見て記録しているため。

動作確認の結果（ランダムに行動を選ぶだけの実験）：
- 20エピソード記録して、`episode`番号が1つずつ正しく増えることを確認
- 鍵を拾う（`pickup`）→`carrying`が`"key"`になる、を確認
- ドアを開ける（`toggle`）→`door_state`が`locked`→`open`に変わる、を確認
- ランダムに動くだけではゴールに着けない（20回中0回）。鍵を拾ってドアを開けてゴールまで
  行くのは単純なランダム行動では難しい、という想定どおりの結果

## まだできないこと（未実装）

- AIの学習（PPOでDoorKeyを解けるように学習する`train_agent.py`）
- 学習済みAIにプレイさせてログを取る（`run_agent.py`）
- 人間にプレイしてもらってログを取る（`play_human.py`）
- 記録したログから統計（行動回数・エピソード長など）を計算する（`analyze_stats.py`）
- LLMにログを見せて人間/AIの違いを聞く（`ask_llm.py`）。Gemini APIキーもまだ未設定

## 動かすための準備

- このフォルダ内の`venv`（Python 3.12）に`minigrid` / `gymnasium` / `stable-baselines3` / `torch`が
  インストール済み（バージョンは`requirements.txt`に記録）
- 注意：このPCではWindowsの「Smart App Control」が最新版torchの未署名DLLをブロックするため、
  `torch==2.8.0`に固定している（詳細は`notes.md`）
- `GEMINI_API_KEY`はまだ設定していない（LLMを使うフェーズ5で必要になる）

## 進み具合

計画全体（フェーズ0〜6）の進捗は`claude.md`の「進行状況」チェックリストを参照。
現在：フェーズ0〜2（環境構築・仕様確認・ログ記録の土台）が完了、フェーズ3（AIの学習）が次。
