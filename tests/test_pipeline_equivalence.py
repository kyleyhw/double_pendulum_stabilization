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

Platform dependence
-------------------
Bit patterns depend on the platform's libm (``sin``/``cos``) and LAPACK
build, not only on the code: the original hashes do not reproduce on
x86-64 Linux even with the locked numpy/gymnasium versions. So there are
two gates:

* **Reference gate (portable, always on).** The trajectory must match
  ``tests/data/pipeline_reference.npz`` to ``atol = 1e-10``. Cross-platform
  rounding differences are ~1e-15 here; any physics change is orders of
  magnitude larger.
* **Bit gate (per platform).** A fingerprint of the platform's
  ``sin``/``cos``/``solve`` output selects the expected hashes from
  :data:`PLATFORM_HASHES`. On an unrecognised platform the test passes if
  the hash equals the original baseline and is otherwise *skipped* with the
  fingerprint and hashes to register (run this file directly to print them).
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

# Platform fingerprint -> (double RK4, single RK4, double semi-implicit) hashes.
PLATFORM_HASHES: dict[str, tuple[str, str, str]] = {
    # x86-64 Linux, numpy 2.3.5 / 2.4.6 (OpenBLAS), recorded 2026-09-26.
    "123b013908082a0e": (
        "c923877c1e51d0c1f21d26376b8ad87ac8095de83e14a8abe97a3603f3f85d90",
        "f944a626720e81bb78a9a4224c99f45e45568f3a6bc1dc9c65dee7a97cd22d78",
        "bc873b7155745712a18feb03b0d24332844348cdc75c65b4554c077d364c98da",
    ),
}
LEGACY_HASHES: tuple[str, str, str] = (
    BASELINE_HASH_DOUBLE_RK4, BASELINE_HASH_SINGLE_RK4, BASELINE_HASH_DOUBLE_SI,
)
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "data", "pipeline_reference.npz")
REFERENCE_ATOL: float = 1e-10


def platform_fingerprint() -> str:
    """Short hash of this platform's elementary-function and 3x3-solve rounding."""
    x = np.linspace(-20.0, 20.0, 20001)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 3, 3)) + 3.0 * np.eye(3)
    b = rng.normal(size=(64, 3))
    parts = [np.sin(x), np.cos(x), np.linalg.solve(A[0], b[0])]
    parts += [np.linalg.solve(A[i], b[i]) for i in range(1, 64)]
    return hashlib.sha256(b"".join(p.tobytes() for p in parts)).hexdigest()[:16]


def _trajectory(env, *, n_steps: int = N_STEPS, seed: int = RNG_SEED) -> np.ndarray:
    env.reset(seed=seed)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    states = np.empty((n_steps, env.state.shape[0]), dtype=np.float64)
    for t in range(n_steps):
        env.step(zero)
        states[t] = np.asarray(env.state, dtype=np.float64)
    return states


def _envs() -> dict[str, object]:
    return {
        "double_rk4": DoublePendulumCartEnv(control_strategy=ForceControl(), integrator="rk4"),
        "single_rk4": SinglePendulumCartEnv(control_strategy=ForceControl(), integrator="rk4"),
        "double_si": DoublePendulumCartEnv(control_strategy=ForceControl(),
                                           integrator="semi_implicit"),
    }


_KEYS = ("double_rk4", "single_rk4", "double_si")


def _expected_hash(key: str, got: str) -> str | None:
    """Expected hash on this platform, or ``None`` if the platform is unknown and
    the hash does not match the legacy baseline (caller should skip)."""
    i = _KEYS.index(key)
    fp = platform_fingerprint()
    if fp in PLATFORM_HASHES:
        return PLATFORM_HASHES[fp][i]
    return LEGACY_HASHES[i] if got == LEGACY_HASHES[i] else None


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

    def _check_bits(self, key: str) -> None:
        h = _trajectory_hash(_envs()[key])
        expected = _expected_hash(key, h)
        if expected is None:
            self.skipTest(f"Unrecognised platform (fingerprint {platform_fingerprint()}); "
                          f"{key} hash {h}. Register it in PLATFORM_HASHES to enable the bit "
                          "gate here; the portable reference gate still applies.")
        self.assertEqual(
            h, expected,
            f"{key} trajectory hash drifted from this platform's baseline.\n"
            f"  expected: {expected}\n"
            f"  got:      {h}\n"
            "Physics has changed — this candidate is not bit-equivalent."
        )

    def _check_reference(self, key: str) -> None:
        ref = np.load(REFERENCE_PATH)[key]
        np.testing.assert_allclose(
            _trajectory(_envs()[key]), ref, rtol=0.0, atol=REFERENCE_ATOL,
            err_msg=f"{key} trajectory differs from tests/data/pipeline_reference.npz "
                    "beyond platform rounding — physics has changed.")

    def test_double_pendulum_rk4_bit_identical(self) -> None:
        self._check_bits("double_rk4")

    def test_single_pendulum_rk4_bit_identical(self) -> None:
        self._check_bits("single_rk4")

    def test_double_pendulum_semi_implicit_bit_identical(self) -> None:
        self._check_bits("double_si")

    def test_double_pendulum_rk4_matches_reference(self) -> None:
        self._check_reference("double_rk4")

    def test_single_pendulum_rk4_matches_reference(self) -> None:
        self._check_reference("single_rk4")

    def test_double_pendulum_semi_implicit_matches_reference(self) -> None:
        self._check_reference("double_si")

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
    if "--register" in sys.argv:
        # Print this platform's fingerprint and hashes for PLATFORM_HASHES.
        envs = _envs()
        print(f'    "{platform_fingerprint()}": (')
        for k in _KEYS:
            print(f'        "{_trajectory_hash(envs[k])}",')
        print("    ),")
    else:
        unittest.main(verbosity=2)
