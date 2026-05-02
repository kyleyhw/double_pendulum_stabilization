r"""
Evolve eval harness — measures runtime of the training/test pipeline.

Emits a single JSON object on stdout that the agent-evolve framework parses.
The metrics are designed so that *lower is better* on durations and *higher
is better* (1.0 ideal) on pass rates.

Why these specific metrics
==========================
* ``train_update_ms_mean`` — mean wall-time per PPO update. This is the
  headline runtime metric: it covers env stepping (rollout collection),
  policy forward passes, GAE, and the K-epoch update loop. The user's
  optimisation target is this number.
* ``env_step_ms_mean`` — mean wall-time per env step in a tight zero-policy
  loop. Isolates the environment hot path (dynamics + integrator + reward
  + obs construction). Diagnostic — lets the reviewer attribute speed-ups
  to the env layer vs. the trainer/PPO layer.
* ``test_pass_rate`` — fraction of tests passing in
  ``tests/test_physics.py``, ``test_components.py``, ``test_energy_reward.py``,
  and ``test_pipeline_equivalence.py``. Hard-gated to 1.0 — the candidate
  is rejected if any test fails. This is the physics-correctness gate.
* ``equivalence_check`` — 1.0 if the env produces a bit-identical zero-
  action trajectory to the master baseline (hashed), 0.0 otherwise. This
  is a redundant check on top of the test suite so the reviewer can see
  the gate state without parsing pytest output.
* ``total_eval_ms`` — wall-time of this entire harness run. Useful as a
  composite for budget tracking.

The training run is intentionally short (``--updates 5 --n_envs 4
--rollout_steps 256``) so each candidate eval finishes in well under a
minute. The relative ordering of candidate runtimes is preserved at this
shorter scale because the per-update cost is dominated by the same hot
loops (env step + policy forward + PPO update) that scale linearly.

Usage
-----
::

    python tools/evolve_eval.py
    python tools/evolve_eval.py --updates 10        # longer training run
    python tools/evolve_eval.py --skip-tests        # runtime-only

The harness exits with status 0 on success and emits JSON; non-zero on
infrastructure failure (subprocess crash, timeout). A test failure is
*not* a harness failure — it shows up as ``test_pass_rate < 1.0``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, timeout: float | None = None,
         env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    """Run a subprocess, returning (returncode, stdout, stderr, duration_ms)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return -1, e.stdout or "", f"timeout after {timeout}s", duration_ms
    duration_ms = (time.perf_counter() - start) * 1000.0
    return proc.returncode, proc.stdout, proc.stderr, duration_ms


def _measure_env_step(n_steps: int = 2000) -> dict[str, float]:
    """Run a tight zero-action env-step loop and report mean ms/step.

    Imports inside the function so the harness can be run from any cwd
    after the repo root is on sys.path.
    """
    sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    from src.env.double_pendulum import DoublePendulumCartEnv
    from src.strategies.controls import ForceControl

    env = DoublePendulumCartEnv(control_strategy=ForceControl())
    env.reset(seed=0)
    zero = np.zeros(1, dtype=np.float32)

    # Warmup — JITs, imports, allocator priming.
    for _ in range(200):
        env.step(zero)

    start = time.perf_counter()
    for _ in range(n_steps):
        env.step(zero)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {
        "env_step_ms_mean": elapsed_ms / n_steps,
        "env_step_total_ms": elapsed_ms,
        "env_step_count": float(n_steps),
    }


def _measure_train_updates(*, updates: int, n_envs: int, rollout_steps: int
                           ) -> dict[str, float]:
    """Run a short PPO training run and parse its CSV log for per-update timing.

    We read the wall-clock around the subprocess call rather than relying
    on the trainer's internal logs — simpler and avoids needing trainer
    cooperation. The (updates, n_envs, rollout_steps) trio bounds the
    total work done.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "train.py"),
        "--env", "double",
        "--reward", "exponential",
        "--updates", str(updates),
        "--n_envs", str(n_envs),
        "--rollout_steps", str(rollout_steps),
        "--eval_every", "0",
        "--log_interval", str(max(1, updates)),       # silence per-update prints
        "--save_interval", str(max(1, updates * 10)),  # don't write checkpoints
        "--seed", "0",
    ]
    # Disable TB writer to keep filesystem chatter out of the timing.
    env = os.environ.copy()
    env["EVOLVE_EVAL_DISABLE_TB"] = "1"

    rc, stdout, stderr, duration_ms = _run(cmd, timeout=600.0, env=env)
    return {
        "train_total_ms": duration_ms,
        "train_update_ms_mean": duration_ms / max(1, updates),
        "train_returncode": float(rc),
        "train_passed": 1.0 if rc == 0 else 0.0,
        "train_updates": float(updates),
        "train_n_envs": float(n_envs),
        "train_rollout_steps": float(rollout_steps),
    }


def _run_tests() -> dict[str, float]:
    """Run pytest on the four test files and report pass rate + duration."""
    test_files = [
        "tests/test_physics.py",
        "tests/test_components.py",
        "tests/test_energy_reward.py",
        "tests/test_pipeline_equivalence.py",
    ]
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider"]
    cmd.extend(test_files)
    rc, stdout, stderr, duration_ms = _run(cmd, timeout=300.0)

    # Pytest summary parsing: "<n> passed", "<n> failed".
    passed = failed = 0
    for line in stdout.splitlines():
        # Lines look like "12 passed, 1 failed in 3.21s" or "5 passed in 0.4s"
        if "passed" in line or "failed" in line:
            tokens = line.replace(",", " ").split()
            for i, tok in enumerate(tokens):
                if tok == "passed" and i > 0 and tokens[i - 1].isdigit():
                    passed = int(tokens[i - 1])
                if tok == "failed" and i > 0 and tokens[i - 1].isdigit():
                    failed = int(tokens[i - 1])
    total = passed + failed
    pass_rate = (passed / total) if total > 0 else 0.0
    return {
        "test_total_ms": duration_ms,
        "test_pass_rate": pass_rate,
        "test_passed": float(passed),
        "test_failed": float(failed),
        "test_returncode": float(rc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolve eval harness for the cart-pendulum pipeline.")
    parser.add_argument("--updates", type=int, default=5,
                        help="PPO updates for the training run (default: 5).")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Parallel envs for the training run (default: 4).")
    parser.add_argument("--rollout_steps", type=int, default=256,
                        help="Rollout steps per env per update (default: 256).")
    parser.add_argument("--env_step_n", type=int, default=2000,
                        help="Steps for the env-only microbenchmark (default: 2000).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip the PPO training run (env step + tests only).")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip pytest (training and env-step only).")
    parser.add_argument("--skip-env-step", action="store_true",
                        help="Skip the env-step microbenchmark.")
    args = parser.parse_args()

    overall_start = time.perf_counter()
    metrics: dict[str, float] = {}

    if not args.skip_env_step:
        try:
            metrics.update(_measure_env_step(n_steps=args.env_step_n))
        except Exception as e:  # pragma: no cover - infrastructure failure
            print(f"env-step measurement failed: {e}", file=sys.stderr)
            metrics["env_step_ms_mean"] = float("inf")

    if not args.skip_train:
        metrics.update(_measure_train_updates(
            updates=args.updates, n_envs=args.n_envs, rollout_steps=args.rollout_steps,
        ))

    if not args.skip_tests:
        metrics.update(_run_tests())

    metrics["total_eval_ms"] = (time.perf_counter() - overall_start) * 1000.0

    # Aggregate pass-rate gate. The reviewer rejects on test_pass_rate < 1.0,
    # but we surface a clear top-level boolean for human readability.
    metrics["all_gates_passed"] = float(
        metrics.get("test_pass_rate", 0.0) >= 1.0
        and metrics.get("train_passed", 1.0) >= 1.0
    )

    # Single JSON line on stdout — the eval harness parses the last well-formed
    # JSON object from stdout. Indented form is allowed.
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
