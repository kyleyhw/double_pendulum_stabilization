r"""
Honest evaluation of a Phase S policy (``src/train_balance.py``) on the real
``DoublePendulumCartEnv`` at full difficulty (:math:`\delta = 1`, wind
:math:`\sigma_w = 1` N), with ``ForceControl(max_force=F_max)`` and the
policy's action repeat.

Two test sets, 20 s episodes, the Phase O metrics from
``tools/eval_swingup.py``:

* ``basin``   — starts drawn from the training basin at radius ``--radius``
  (angles :math:`\pm r`, rates :math:`\pm 2r`, cart :math:`\pm 0.5`), i.e.
  the NEXT_STEPS Option B milestone (≥ 95 % terminal-1 s strict at 0.25 rad);
* ``hanging`` — the env's own ``down`` reset (full swing-up from hanging).

Usage::

    python tools/eval_balance.py --model logs/phaseS_seed0_final.pth
    python tools/eval_balance.py --model ... --report docs/reports/phaseS_balance.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.sac import GaussianPolicy  # noqa: E402
from src.control.switching import NAMES  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402
from src.train_balance import make_obs  # noqa: E402
from tools.eval_swingup import run_episode  # noqa: E402


class PolicyController:
    """Deterministic SAC actor behind the ``reset`` / ``action`` controller interface,
    re-deciding every ``frame_skip`` simulator steps (as in training)."""

    def __init__(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        self.max_force = float(meta["max_force"])
        self.frame_skip = int(meta["frame_skip"])
        self.multi_goal = bool(meta.get("multi_goal", False))
        self.goal = NAMES.index("UU")
        self.actor = GaussianPolicy(8 + (len(NAMES) if self.multi_goal else 0), 1,
                                    int(meta["hidden_dim"]))
        self.actor.load_state_dict(payload["actor"])
        self.actor.eval()
        self.curriculum = payload.get("curriculum", {})
        self.reset()

    def reset(self, state: np.ndarray | None = None) -> None:
        self._k = 0
        self._a = 0.0

    def force(self, state: np.ndarray) -> float:
        if self._k % self.frame_skip == 0:
            with torch.no_grad():
                obs = make_obs(state, np.array([self.goal]) if self.multi_goal else None)
                a, _ = self.actor.sample(torch.as_tensor(obs), deterministic=True)
            self._a = float(a[0, 0])
        self._k += 1
        return self._a * self.max_force

    def action(self, state: np.ndarray, max_force: float) -> np.ndarray:
        return np.array([self.force(state) / max_force], dtype=np.float32)


class PolicySwitcher(PolicyController):
    """A goal-conditioned policy behind :class:`SwitchingController`'s interface
    (``reset(src)``, ``request(dst)``, ``settled``, ``current``, ``lib``), so
    ``tools/eval_switching.py`` can score it the same way."""

    def __init__(self, path: str, lib: dict) -> None:
        super().__init__(path)
        if not self.multi_goal:
            raise ValueError(f"{path} is not a goal-conditioned policy")
        self.lib = lib
        self.phase = "hold"

    def reset(self, initial: str | np.ndarray | None = "DD") -> None:
        super().reset()
        if isinstance(initial, str):
            self.current = self.target = initial
            self.goal = NAMES.index(initial)

    def request(self, target: str) -> None:
        self.current = self.target = target
        self.goal = NAMES.index(target)

    @property
    def settled(self) -> bool:
        return True


def basin_state(rng: np.random.Generator, r: float) -> np.ndarray:
    s = np.zeros(6)
    s[0] = rng.uniform(-0.5, 0.5)
    s[1:3] = np.pi + rng.uniform(-r, r, 2)
    s[3] = rng.uniform(-0.5, 0.5)
    s[4:6] = rng.uniform(-2 * r, 2 * r, 2)
    return s


def summarize(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "survived": float(np.mean([r["survived"] for r in rows])),
        "in_soft": float(np.mean([r["in_soft"] for r in rows])),
        "terminal_strict": float(np.mean([r["terminal_strict"] for r in rows])),
        "full_terminal": int(sum(r["terminal_strict"] == 1.0 for r in rows)),
        "sustained": float(np.mean([r["max_sustained"] for r in rows])),
        "ss_err": np.mean([r["ss_err"] for r in rows], axis=0),
        "max_force": float(max(r["max_force"] for r in rows)),
    }


def evaluate(model: str, episodes: int, radius: float, horizon: float = 20.0,
             seed: int = 0) -> dict[str, dict]:
    ctrl = PolicyController(model)
    env = DoublePendulumCartEnv(control_strategy=ForceControl(max_force=ctrl.max_force))
    env.set_curriculum(1.0)
    rng = np.random.default_rng(seed)
    basin = [run_episode(env, ctrl, seed + i, horizon, init_state=basin_state(rng, radius))
             for i in range(episodes)]
    hanging = [run_episode(env, ctrl, seed + i, horizon, reset_mode="down")
               for i in range(episodes)]
    return {"basin": summarize(basin), "hanging": summarize(hanging), "ctrl": ctrl}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--horizon", type=float, default=20.0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    res = evaluate(args.model, args.episodes, args.radius, args.horizon)
    ctrl = res.pop("ctrl")
    rows = [
        ("Survived", lambda s: f"{100 * s['survived']:.1f} %"),
        ("Stayed within soft cart bound", lambda s: f"{100 * s['in_soft']:.1f} %"),
        ("Terminal-1 s strict (mean)", lambda s: f"{100 * s['terminal_strict']:.1f} %"),
        ("Episodes with terminal-1 s strict = 100 %",
         lambda s: f"{s['full_terminal']} / {s['episodes']}"),
        ("Max sustained strict (mean)", lambda s: f"{s['sustained']:.2f} s"),
        ("Final-second error P1 / P2", lambda s: f"{s['ss_err'][0]:.2f}° / {s['ss_err'][1]:.2f}°"),
        ("Peak cart force", lambda s: f"{s['max_force']:.0f} N"),
    ]
    header = f"| Metric | Basin r = {args.radius:g} rad | From hanging |"
    table = [header, "|---|---:|---:|"]
    table += [f"| {k} | {f(res['basin'])} | {f(res['hanging'])} |" for k, f in rows]
    print("\n".join(table))
    if args.report:
        lines = [
            "# Phase S policy evaluation (δ = 1)",
            "",
            f"Model `{args.model}` (curriculum reached: {ctrl.curriculum}); "
            f"F_max = {ctrl.max_force:g} N, action repeat {ctrl.frame_skip} "
            f"({ctrl.frame_skip * 5} ms). {args.episodes} episodes × {args.horizon:g} s per "
            "column on `DoublePendulumCartEnv` (wind σ = 1 N), deterministic policy.",
            "",
            *table,
            "",
            f"Command: `python tools/eval_balance.py --model {args.model} --episodes "
            f"{args.episodes} --radius {args.radius:g}`",
            "",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(lines))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
