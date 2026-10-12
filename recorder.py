"""ログ記録用の Gymnasium Wrapper。

MiniGrid の1エピソードを「1行1step」の JSONL 形式で記録する。
AI (source="agent") と人間 (source="human") で同じクラス・同じ項目を使うことで、
後段で LLM に見せるときにフォーマットの違いが紛れ込まないようにする。

設計上の注意点：
- ログに書く値は、内側に FullyObsWrapper などの ObservationWrapper がかかっていても
  常に `env.unwrapped` のグリッド・エージェント状態から直接読む。つまり「ポリシーに何を
  見せるか（観測空間の選択）」と「ログに何を残すか」を分離している。
- 生のグリッド画像（grid.encode() の配列）はログに保存しない。seed と行動列から
  後で再現できるため、JSONL を軽く・読みやすく保つことを優先した
  （ユーザー確認済み：2026-10-12）。
- 行動回数や往復などの統計はここでは計算しない。1stepごとの生データだけを記録し、
  集計は analyze_stats.py（フェーズ4）側でコードとして行う。LLMには数えさせない方針
  （CLAUDE.md参照）。
- `t`（記録時刻のUNIX時刻）は記録するが、AIの t は計算時間であり人間の反応時間とは
  比較できないため、最初の人間/AI比較には使わない（CLAUDE.md参照）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions


def _to_builtin(value: Any) -> Any:
    """numpyのスカラー/配列をJSONに書き込めるPython組み込み型に変換する。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(v) for v in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_to_builtin(v) for v in value]
    return value


def _find_door(env) -> dict | None:
    """グリッド内のドアを探して位置と状態を返す。

    DoorKey環境にはドアが1枚しかない前提。見つからない場合は None。
    状態の判定順序は Door.encode() の仕様（open > locked > closed）に合わせている。
    """
    for x in range(env.grid.width):
        for y in range(env.grid.height):
            cell = env.grid.get(x, y)
            if cell is not None and cell.type == "door":
                if cell.is_open:
                    state = "open"
                elif cell.is_locked:
                    state = "locked"
                else:
                    state = "closed"
                return {"x": x, "y": y, "state": state}
    return None


class EpisodeRecorder(gym.Wrapper):
    """1step・1エピソードをJSONLファイルに記録するラッパー。

    使い方:
        env = gym.make("MiniGrid-DoorKey-6x6-v0")
        env = FullyObsWrapper(env)  # 観測を全体観測にする場合はここでかける
        env = EpisodeRecorder(env, "logs/agent/run.jsonl", source="agent")

        obs, info = env.reset(seed=0)   # ここで step=0 の行（初期状態）を1行書く
        obs, reward, terminated, truncated, info = env.step(action)  # step=1 の行を書く
        ...
        env.close()  # ファイルをきちんと閉じる
    """

    def __init__(
        self,
        env: gym.Env,
        log_path: str | Path,
        source: str,
        append: bool = True,
    ):
        super().__init__(env)
        if source not in ("agent", "human"):
            raise ValueError(f"source must be 'agent' or 'human', got {source!r}")
        self.source = source
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.log_path, "a" if append else "w", encoding="utf-8")
        self._episode = 0
        self._step = 0
        self._seed: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode += 1
        self._step = 0
        self._seed = seed
        self._write_row(
            action=None,
            action_name=None,
            reward=None,
            terminated=False,
            truncated=False,
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        action_idx = int(action)
        self._write_row(
            action=action_idx,
            action_name=Actions(action_idx).name,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return obs, reward, terminated, truncated, info

    def _write_row(
        self,
        *,
        action: int | None,
        action_name: str | None,
        reward: float | None,
        terminated: bool,
        truncated: bool,
    ) -> None:
        env = self.env.unwrapped
        pos = _to_builtin(env.agent_pos)
        carrying = env.carrying
        door = _find_door(env)

        row = {
            "source": self.source,
            "episode": self._episode,
            "seed": self._seed,
            "step": self._step,
            "pos_x": pos[0],
            "pos_y": pos[1],
            "direction": int(env.agent_dir),
            "action": action,
            "action_name": action_name,
            "carrying": carrying.type if carrying is not None else None,
            "carrying_color": carrying.color if carrying is not None else None,
            "door_x": door["x"] if door else None,
            "door_y": door["y"] if door else None,
            "door_state": door["state"] if door else None,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": bool(terminated or truncated),
            "t": time.time(),
        }
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
        return self.env.close()
