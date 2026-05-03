r"""
Algorithm-evolve eval harness — runs a short PPO training fragment with a
candidate-supplied hyperparameter config, evaluates the resulting policy,
and emits JSON metrics for the agent-evolve framework to score against.

Why this exists separately from `tools/evolve_eval.py`
=====================================================
The runtime harness measures **wall-time per update**; this harness measures
**policy quality after a training fragment**. They have orthogonal roles:

* Runtime harness gates `mode: runtime` candidates (PR #1, PR #2). Their
  changes must produce a bit-identical physics evolution and ideally faster
  wall-time. The harness runs 5 PPO updates and times them — it doesn't
  care what reward the policy is converging toward.

* This harness gates `mode: algorithm` candidates. Their changes must
  produce a *better policy* on the headline task: stabilise the cart-double-
  pendulum at upright. We measure that by training from scratch for ``N``
  updates with the candidate's HP config, then running a deterministic
  evaluation at the difficulty the agent reached.

Metrics
-------
Primary: ``strict_success_rate`` (% time both poles within :math:`10^\circ` of
upright in deterministic eval). The campaign log shows a ~6.5% ceiling under
Phase I; this harness's job is to find a config that breaks it.

Secondary:

* ``loose_success_rate`` — same but :math:`20^\circ`. More forgiving, more
  sensitive to early progress.
* ``peak_difficulty`` — highest curriculum :math:`\delta` reached during
  training. Reflects how aggressively the ratchet fired.
* ``final_difficulty`` — :math:`\delta` at end of training.
* ``final_eval_reward`` — mean episode reward in the deterministic eval.
* ``wall_time_min`` — wall-time of the training fragment in minutes.
* ``physics_tests_pass`` — 1.0 if all 21 physics-anchor tests pass on the
  candidate's checkout, 0.0 otherwise. **Hard gate**: any candidate that
  breaks physics is rejected regardless of policy quality.
* ``train_completed`` — 1.0 if the training subprocess returned 0, 0.0 otherwise.

Usage
-----
::

    python tools/algorithm_eval.py --config configs/round1_c1.yaml
    python tools/algorithm_eval.py --updates 500 --seed 42 \
        --reward hybrid --threshold_curve concave \
        --lr 3e-4 --target_kl 0.015 --value_clip 0.2

The config file is a flat YAML mapping of CLI flag → value (without the
leading ``--``). Any flag accepted by ``src/train.py`` is allowed. Defaults
mirror the Phase H3/I config that produced the current best.

Eval-fragment design
--------------------
Defaults run a 500-update training fragment at ``--n_envs 8 --rollout_steps
512``. At ~1.34 s/update on the c6 pipeline, this is ~11 minutes per
candidate.

The training is launched **from scratch** (no ``--load``) so candidates
compete on equal footing; this is honest comparison, not a leak from a
parent checkpoint that may have been tuned to a different reward shape.

The deterministic eval runs at the *peak* difficulty reached, which is
read from the training CSV log (``Difficulty`` column max). This avoids
penalising configs that ratcheted briefly then drifted back: we measure
the policy at the point where it was best, with the curriculum the agent
*proved* it could handle.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults mirror the Phase H3/I config (the strongest known baseline as of
# the campaign log). Override any subset via --config or CLI flags.
DEFAULTS: dict[str, Any] = {
    "env": "double",
    "control": "velocity",
    "reward": "hybrid",
    "threshold_curve": "concave",
    "integrator": "rk4",
    "n_envs": 8,
    "rollout_steps": 512,
    # Round-1 lesson: 500 from-scratch updates can't separate configs (too
    # short for anyone to break swing-up). Round 2+ defaults to continuing
    # from the Phase L best for 200 updates — sharp signal on whether HP
    # perturbations push past the existing 4-7% ceiling.
    "updates": 200,
    "episode_steps": 4000,
    "load": "logs/ppo_double_velocity_hybrid_20260503_000503_best.pth",
    "start_difficulty": 0.465,
    "obs_norm_freeze_steps": 0,
    "lr": 3e-4,
    "lr_final_frac": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "eps_clip": 0.2,
    "clip_final_frac": 0.5,
    "k_epochs": 10,
    "entropy_coef": 0.01,
    "ent_final_frac": 0.1,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "init_log_std": -0.5,
    "hidden_dim": 256,
    "minibatch_size": 256,
    "target_kl": 0.015,
    "value_clip": 0.2,
    "ratchet_min": 0.005,
    "ratchet_max": 0.05,
    "window": 20,
    "obs_norm_freeze_steps": -1,
    "best_fallback_threshold": 0.5,
    "best_fallback_patience": 3,
    "log_interval": 50,
    "save_interval": 100,
    "eval_every": 0,    # we run a single deterministic eval at the end
    "seed": 0,
}


def _load_config(path: str | None) -> dict[str, Any]:
    """Load a YAML config file (flat mapping of flag-name → value)."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config {p} must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def _build_train_argv(config: dict[str, Any]) -> list[str]:
    """Translate a flat config dict into the CLI argv for ``src/train.py``."""
    argv = [sys.executable, str(REPO_ROOT / "src" / "train.py")]
    for k, v in config.items():
        # Boolean flags are presence-only.
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k}")
            continue
        argv.extend([f"--{k}", str(v)])
    return argv


def _run_subprocess(argv: list[str], timeout: float | None,
                    env: dict[str, str] | None = None
                    ) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv, cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", f"timeout after {timeout}s", time.perf_counter() - start
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def _find_run_artifacts(stdout: str) -> tuple[str | None, str | None]:
    """Parse train.py stdout for the run name + final / best checkpoint paths."""
    run_name: str | None = None
    best_path: str | None = None
    final_path: str | None = None
    for line in stdout.splitlines():
        m = re.search(r"\[run\]\s+(\S+)", line)
        if m:
            run_name = m.group(1)
        m = re.search(r"ppo_(\S+)_best\.pth", line)
        if m:
            best_path = "logs/ppo_" + m.group(1) + "_best.pth"
        m = re.search(r"final checkpoint:\s+(\S+)", line)
        if m:
            final_path = m.group(1).replace("\\", "/")
    # Pick the best snapshot when present (more representative of policy
    # quality than the final, which may have drifted under continued PPO).
    return run_name, (best_path or final_path)


def _read_csv_metrics(run_name: str | None) -> dict[str, float]:
    """Read the training CSV log for peak/final difficulty."""
    if run_name is None:
        return {}
    csv_path = REPO_ROOT / "logs" / f"training_log_{run_name}.csv"
    if not csv_path.exists():
        return {}
    peak = 0.0
    final = 0.0
    final_reward = 0.0
    last_update = 0
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        try:
            i_diff = header.index("Difficulty")
            i_reward = header.index("Reward")
            i_update = header.index("Update")
        except ValueError:
            return {}
        for line in f:
            row = line.strip().split(",")
            if len(row) <= max(i_diff, i_reward, i_update):
                continue
            try:
                d = float(row[i_diff])
                r = float(row[i_reward])
                u = int(row[i_update])
            except ValueError:
                continue
            if d > peak:
                peak = d
            final = d
            final_reward = r
            last_update = u
    return {
        "peak_difficulty": peak,
        "final_difficulty": final,
        "final_window_reward": final_reward,
        "updates_completed": float(last_update),
    }


def _run_diagnostic_eval(checkpoint: str, difficulty: float,
                         episodes: int = 30) -> dict[str, float]:
    """Run evaluate_diagnostics.py and parse its markdown report."""
    if not checkpoint or not (REPO_ROOT / checkpoint).exists():
        return {"strict_success_rate": 0.0, "loose_success_rate": 0.0,
                "diag_eval_failed": 1.0}
    out_path = REPO_ROOT / "docs" / "reports" / "_algorithm_eval_tmp.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, str(REPO_ROOT / "src" / "evaluate_diagnostics.py"),
        "--model", checkpoint,
        "--episodes", str(episodes),
        "--difficulty", f"{difficulty:.4f}",
        "--output", str(out_path),
    ]
    rc, stdout, stderr, _ = _run_subprocess(argv, timeout=300.0)
    if rc != 0 or not out_path.exists():
        return {"strict_success_rate": 0.0, "loose_success_rate": 0.0,
                "diag_eval_failed": 1.0,
                "diag_eval_stderr": 1.0 if stderr else 0.0}
    text = out_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, float] = {"diag_eval_failed": 0.0}
    for label, key in [
        ("Strict Success Rate", "strict_success_rate"),
        ("Loose Success Rate", "loose_success_rate"),
        ("Avg Reward", "diag_eval_mean_reward"),
        ("Avg Length", "diag_eval_mean_length"),
        ("Steady-State Error \\(P1\\)", "sse_pole_1_deg"),
        ("Steady-State Error \\(P2\\)", "sse_pole_2_deg"),
        ("Control Effort", "control_effort"),
    ]:
        # Markdown is `| **Strict Success Rate** | **4.4%** | ...`.
        m = re.search(rf"\*\*{label}\*\*\s*\|\s*\*?\*?(-?\d+\.?\d*)%?", text)
        if m:
            val = float(m.group(1))
            if "rate" in key:
                val /= 100.0  # convert percent → fraction
            metrics[key] = val
    return metrics


def _run_physics_tests() -> dict[str, float]:
    """Run the four physics-anchor test files; return pass-rate gate."""
    argv = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_physics.py",
        "tests/test_components.py",
        "tests/test_energy_reward.py",
        "tests/test_pipeline_equivalence.py",
    ]
    rc, stdout, stderr, elapsed = _run_subprocess(argv, timeout=300.0)
    passed = failed = 0
    for line in stdout.splitlines():
        if "passed" in line or "failed" in line:
            tokens = line.replace(",", " ").split()
            for i, tok in enumerate(tokens):
                if tok == "passed" and i > 0 and tokens[i - 1].isdigit():
                    passed = int(tokens[i - 1])
                if tok == "failed" and i > 0 and tokens[i - 1].isdigit():
                    failed = int(tokens[i - 1])
    total = passed + failed
    return {
        "physics_tests_pass": (passed / total) if total > 0 else 0.0,
        "physics_test_passed": float(passed),
        "physics_test_failed": float(failed),
        "physics_test_returncode": float(rc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (flat flag → value mapping).")
    parser.add_argument("--diag_episodes", type=int, default=30,
                        help="Episodes for the deterministic eval (default 30).")
    parser.add_argument("--train_timeout_min", type=float, default=45.0,
                        help="Timeout for the training subprocess.")
    parser.add_argument("--skip_physics_gate", action="store_true",
                        help="Skip the physics-anchor test gate (debug only).")
    args, passthrough = parser.parse_known_args()

    config = dict(DEFAULTS)
    config.update(_load_config(args.config))
    # Allow ad-hoc CLI overrides: --reward hybrid --target_kl 0.01 …
    for tok_idx in range(0, len(passthrough), 2):
        if tok_idx + 1 >= len(passthrough):
            break
        key = passthrough[tok_idx]
        if not key.startswith("--"):
            continue
        config[key[2:]] = _coerce(passthrough[tok_idx + 1])

    metrics: dict[str, float] = {}

    if not args.skip_physics_gate:
        metrics.update(_run_physics_tests())
        # Hard gate: if physics broken, do not waste minutes on a training run.
        if metrics.get("physics_tests_pass", 0.0) < 1.0:
            metrics["train_completed"] = 0.0
            metrics["strict_success_rate"] = 0.0
            metrics["loose_success_rate"] = 0.0
            metrics["wall_time_min"] = 0.0
            print(json.dumps(metrics, indent=2, sort_keys=True))
            return 0

    argv = _build_train_argv(config)
    rc, stdout, stderr, train_seconds = _run_subprocess(
        argv, timeout=args.train_timeout_min * 60.0,
    )
    metrics["train_completed"] = 1.0 if rc == 0 else 0.0
    metrics["wall_time_min"] = train_seconds / 60.0

    run_name, checkpoint = _find_run_artifacts(stdout)
    metrics.update(_read_csv_metrics(run_name))

    diag_difficulty = metrics.get("peak_difficulty",
                                  config.get("start_difficulty", 0.0))
    if diag_difficulty <= 0.0:
        diag_difficulty = max(0.05, float(config.get("start_difficulty", 0.05)))

    metrics.update(_run_diagnostic_eval(
        checkpoint or "", difficulty=float(diag_difficulty),
        episodes=args.diag_episodes,
    ))
    metrics["diag_eval_difficulty"] = float(diag_difficulty)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _coerce(s: str) -> Any:
    """Convert a CLI string token to int/float/bool/str."""
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


if __name__ == "__main__":
    sys.exit(main())
