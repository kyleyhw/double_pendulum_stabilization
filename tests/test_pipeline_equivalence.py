r"""
Bit-equivalence test for the env hot path.

Why this exists
===============
Optimisations that touch :py:meth:`_dynamics`, the integrators,
:py:meth:`_get_obs`, or the cart bound logic are *meant* to be physics-
identical to the master baseline — only allocation patterns and
control-flow change. This test asserts that property at the byte level
by hashing a long zero-action trajectory and comparing against a hash
captured on the unoptimised master branch.

Method
------
For each (env_class, integrator) pair:

1. Construct the env with :class:`ForceControl` (action 0 → force 0; this
   isolates the dynamics from any control-strategy logic).
2. Reset with the fixed seed ``RNG_SEED = 42``.
3. Step the env :math:`N = 1000` times with the zero-action vector,
   recording the full internal state :math:`s_t \in \mathbb R^d` at each
   step into a contiguous ``(N, d)`` ``float64`` buffer.
4. Hash the buffer's raw bytes with SHA-256.
5. Assert the hash matches the baseline value below.

The hash is over ``state.tobytes()`` after a deliberate cast to
``float64``. The internal state is already ``float64`` in the current
implementation; the explicit cast guards against an optimisation that
silently downcasts the integrator to ``float32`` — which would change
the dynamics' rounding behaviour and is not allowed.

Updating the baseline
---------------------
If a *physics change* is intentional (e.g. fixing a sign bug, refining
the friction model), regenerate the hashes with::

    python -c "import hashlib, numpy as np, sys; sys.path.insert(0, '.'); \\
        from src.env.double_pendulum import DoublePendulumCartEnv; \\
        from src.strategies.controls import ForceControl; \\
        env = DoublePendulumCartEnv(control_strategy=ForceControl(), integrator='rk4'); \\
        env.reset(seed=42); zero = np.zeros(1, dtype=np.float32); \\
        states = np.empty((1000, 6), dtype=np.float64); \\
        [None for t in range(1000) if (env.step(zero), states.__setitem__(t, env.state))[0]]; \\
        print(hashlib.sha256(states.tobytes()).hexdigest())"

…and replace the constants below. The PR that updates the hashes must
also update :file:`docs/physics_derivation.md` to document the change.

Optimisation candidates that fail this test must NOT be merged. The
baseline hashes are anchored to the master commit prior to the
runtime-pipeline evolution.
"""
from __future__ import annotations

import hashlib
import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.env.double_pendulum import DoublePendulumCartEnv  # noqa: E402
from src.env.single_pendulum import SinglePendulumCartEnv  # noqa: E402
from src.strategies.controls import ForceControl  # noqa: E402

# --- Baseline hashes (computed on master prior to runtime evolution) ------- #
# Seed 42, 1000 zero-action steps, ForceControl. Internal state buffer dtype
# is float64. Source command: tools/evolve_eval.py --skip-train --skip-tests
# (which prints the same numbers on a clean master checkout).
RNG_SEED: int = 42
N_STEPS: int = 1000

BASELINE_HASH_DOUBLE_RK4: str = "e475f8aa8d60539656d9f058e39cf3aa67d56b366fcb7f197375e80b3bcf1ee4"
BASELINE_HASH_SINGLE_RK4: str = "b8994e5b4bb981b78cd243da8a294945611313400b2bb5c86e7f66ca9a0f1992"
BASELINE_HASH_DOUBLE_SI: str = "d1564a02a5b8b14197d50c24a4f61703c9985e19f28271f754af15e794153cbe"


def _trajectory_hash(env, *, n_steps: int = N_STEPS, seed: int = RNG_SEED) -> str:
    """Hash a fixed-seed zero-action trajectory of *env*'s internal state.

    Cast to ``float64`` is explicit so a stealth dtype downgrade (e.g. an
    optimisation that switches the integrator to ``float32`` for speed)
    fails this test even if the trajectory looks "close enough" — it is
    not bit-identical to the baseline and that is the gate.
    """
    env.reset(seed=seed)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    state_dim = env.state.shape[0]
    states = np.empty((n_steps, state_dim), dtype=np.float64)
    for t in range(n_steps):
        env.step(zero)
        states[t] = np.asarray(env.state, dtype=np.float64)
    return hashlib.sha256(states.tobytes()).hexdigest()


class TestPipelineEquivalence(unittest.TestCase):
    r"""Assert physics outputs are bit-identical to the baseline.

    These tests are the gate for the runtime-optimisation evolution.
    A failure means a candidate has changed *what* the env computes,
    not just *how fast* it computes it — physics correctness is broken.
    """

    def test_double_pendulum_rk4_bit_identical(self) -> None:
        env = DoublePendulumCartEnv(control_strategy=ForceControl(), integrator="rk4")
        h = _trajectory_hash(env)
        self.assertEqual(
            h, BASELINE_HASH_DOUBLE_RK4,
            f"Double-pendulum RK4 trajectory hash drifted from baseline.\n"
            f"  expected: {BASELINE_HASH_DOUBLE_RK4}\n"
            f"  got:      {h}\n"
            "Physics has changed — this candidate is not bit-equivalent."
        )

    def test_single_pendulum_rk4_bit_identical(self) -> None:
        env = SinglePendulumCartEnv(control_strategy=ForceControl(), integrator="rk4")
        h = _trajectory_hash(env)
        self.assertEqual(
            h, BASELINE_HASH_SINGLE_RK4,
            f"Single-pendulum RK4 trajectory hash drifted from baseline.\n"
            f"  expected: {BASELINE_HASH_SINGLE_RK4}\n"
            f"  got:      {h}"
        )

    def test_double_pendulum_semi_implicit_bit_identical(self) -> None:
        env = DoublePendulumCartEnv(control_strategy=ForceControl(),
                                    integrator="semi_implicit")
        h = _trajectory_hash(env)
        self.assertEqual(
            h, BASELINE_HASH_DOUBLE_SI,
            f"Double-pendulum semi-implicit Euler trajectory hash drifted from baseline.\n"
            f"  expected: {BASELINE_HASH_DOUBLE_SI}\n"
            f"  got:      {h}"
        )

    def test_observation_constructor_idempotent(self) -> None:
        r"""Two identical resets produce identical observation buffers.

        Catches optimisations that introduce caching bugs in
        :py:meth:`CartPendulumBase._get_obs` (e.g. a returned reference
        to a shared buffer that the next call mutates in place).
        """
        env = DoublePendulumCartEnv(control_strategy=ForceControl())
        obs1, _ = env.reset(seed=RNG_SEED)
        obs2, _ = env.reset(seed=RNG_SEED)
        np.testing.assert_array_equal(
            obs1, obs2,
            err_msg="Observation differs between two identical resets — likely "
                    "a shared-buffer aliasing bug in _get_obs."
        )
        # Mutate obs1 and re-read; obs2 (already captured) must not change.
        obs1[0] = 999.0
        # Direct read of env.state (not _get_obs) — proves the env's internal
        # state is independent of the returned obs buffer.
        self.assertNotEqual(float(env.state[0]), 999.0,
                            "Mutating returned obs corrupts internal env.state — "
                            "the obs buffer must be a copy, not a view.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
