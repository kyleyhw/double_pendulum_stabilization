# Test Report: Energy Conservation

**Date**: 2025-12-01
**Test Suite**: `tests/test_physics.py`
**Runtime**: 0.008s

## 1. Objective (Why)
To verify that the derived Equations of Motion (Lagrangian formulation) and the numerical integrator (RK4) correctly conserve total energy in the absence of non-conservative forces (friction, control input). This confirms the physical correctness of the simulation engine.

## 2. Scope (What)
Tested the `DoublePendulumCartEnv` by initializing it in a high-energy state and simulating for 1 second (50 steps) with zero control force.

## 3. Test Configuration (Specifics)
*   **Input State**: $[x=0, \theta_1=1.0, \theta_2=2.0, \dot{x}=0, \dot{\theta}_1=0, \dot{\theta}_2=0]$.
*   **Rationale**: These angles (approx $57^\circ$ and $115^\circ$) were chosen to ensure the system is in a generic configuration where all coupling terms in the mass matrix and Coriolis vector are active. We avoided simple cases like $\pi/2$ or $0$ to prevent terms from vanishing trivially.
*   **Duration**: 1.0 seconds ($dt=0.02$, 50 steps).

## 4. Results
*   **Initial Energy**: -3.2592 J
*   **Max Deviation**: 0.000097 J
*   **Relative Error**: 0.002991%
*   **Status**: **PASS** (Threshold < 0.1%)

## 5. Failure Analysis
*   *Initial Failure*: The first run failed with a massive energy drift.
*   *Root Cause*: Sign errors in the Gravity Vector ($G(q)$) implementation in `double_pendulum.py`. The potential energy derivative terms were negated incorrectly.
*   *Fix*: Corrected the signs in the `_dynamics` method. The test now passes with high precision.
