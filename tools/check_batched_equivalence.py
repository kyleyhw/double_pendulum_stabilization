r"""
Cross-check: batched vs sequential per-env state evolution.

Goal
====
The trainer's batched fast-path uses :class:`BatchedEnvRunner.step_batch`
to drive N envs in one call. The single-env :py:meth:`step` path is
unchanged and the standard equivalence test (``tests/test_pipeline_equivalence.py``)
already gates that. This script is a redundant manual gate that covers
the *cross-product*: the batched path produces per-row state evolution
that is bit-identical to the unbatched path.

If this fails, the trainer's policy will see slightly different states
than it would have under the sequential code, which would invalidate any
downstream phase that depends on Phase K's exact observed dynamics.

Method
------
1. Construct two parallel sets of N envs, each set seeded identically
   (env i in both sets gets ``seed = base_seed + i``).
2. Take a fixed sequence of 100 random force-control actions per env
   (drawn from a separate, deterministic seed so both paths see the same
   action sequence).
3. For path A: step each env sequentially via :py:meth:`step` and record
   ``env.state`` per env per step into ``traj_A[step, env, :]``.
4. For path B: drive the same envs via :class:`BatchedEnvRunner.step_batch`
   and record into ``traj_B[step, env, :]``.
5. Compare ``traj_A`` and ``traj_B`` element-wise via
   :py:func:`np.array_equal` (bit equality required).

Seeds
-----
* ``BASE_SEED = 42``  — env reset seed, matches the equivalence test.
* ``ACTION_SEED = 7`` — separate stream for the action sequence so the
  envs' own ``np_random`` (used for wind in the random-action regime) is
  independent of the action source.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.cart_pendulum_base import BatchedEnvRunner  # noqa: E402
from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl, VelocityControl  # noqa: E402

BASE_SEED: int = 42
ACTION_SEED: int = 7
N_ENVS: int = 4
N_STEPS: int = 100


def _build_envs(env_cls, control_cls, n: int, base_seed: int) -> list:
    envs = []
    for i in range(n):
        e = env_cls(control_strategy=control_cls(), integrator="rk4", wind_std=0.0)
        e.reset(seed=base_seed + i)
        envs.append(e)
    return envs


def _run_check(env_cls, control_cls, *, with_wind: bool, label: str) -> bool:
    print(f"--- {label} (wind={with_wind}) ---")
    envs_A = _build_envs(env_cls, control_cls, N_ENVS, BASE_SEED)
    envs_B = _build_envs(env_cls, control_cls, N_ENVS, BASE_SEED)

    if with_wind:
        for e in envs_A + envs_B:
            e.set_wind_pinned(0.5)

    state_dim = envs_A[0]._state_dim()
    action_dim = envs_A[0].action_space.shape[0]

    # Deterministic action sequence shared by both paths.
    rng = np.random.default_rng(ACTION_SEED)
    actions = rng.uniform(-1.0, 1.0, size=(N_STEPS, N_ENVS, action_dim)).astype(np.float32)

    # Path A: sequential.
    traj_A = np.empty((N_STEPS, N_ENVS, state_dim), dtype=np.float64)
    for t in range(N_STEPS):
        for i in range(N_ENVS):
            envs_A[i].step(actions[t, i])
            traj_A[t, i] = np.asarray(envs_A[i].state, dtype=np.float64)

    # Path B: batched.
    runner = BatchedEnvRunner(envs_B)
    traj_B = np.empty((N_STEPS, N_ENVS, state_dim), dtype=np.float64)
    for t in range(N_STEPS):
        runner.step_batch(actions[t])
        for i in range(N_ENVS):
            traj_B[t, i] = np.asarray(envs_B[i].state, dtype=np.float64)

    equal = np.array_equal(traj_A, traj_B)
    if equal:
        print(f"  [pass] {N_STEPS} steps x {N_ENVS} envs bit-identical")
    else:
        diff = traj_A - traj_B
        max_abs = float(np.max(np.abs(diff)))
        first_diff_step = int(np.argmax(np.any(diff != 0, axis=(1, 2))))
        print(f"  [fail] max |diff| = {max_abs:.3e}; first diff at step {first_diff_step}")
        print(f"         A[0]: {traj_A[first_diff_step, 0]}")
        print(f"         B[0]: {traj_B[first_diff_step, 0]}")
    return equal


def main() -> int:
    results = []
    results.append(_run_check(DoublePendulumCartEnv, ForceControl,
                              with_wind=False, label="Double / Force / no wind"))
    results.append(_run_check(DoublePendulumCartEnv, ForceControl,
                              with_wind=True, label="Double / Force / wind"))
    results.append(_run_check(DoublePendulumCartEnv, VelocityControl,
                              with_wind=False, label="Double / Velocity / no wind"))
    results.append(_run_check(SinglePendulumCartEnv, ForceControl,
                              with_wind=False, label="Single / Force / no wind"))
    results.append(_run_check(SinglePendulumCartEnv, VelocityControl,
                              with_wind=True, label="Single / Velocity / wind"))

    if all(results):
        print("\nALL CHECKS PASSED")
        return 0
    print(f"\nFAILED ({sum(not r for r in results)}/{len(results)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
