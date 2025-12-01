# Visualization Guide

This document explains how to interpret the real-time visualization of the Double Pendulum system.

## Visualizer Screenshot
![Double Pendulum Simulation](images/visualizer_screenshot.png)

## Color Legend

| Component | Color | Description |
| :--- | :--- | :--- |
| **Cart** | ⚫ Black | The movable base of the system. |
| **Pendulum 1** | 🔵 Blue | The first link attached to the cart. |
| **Pendulum 2** | 🔴 Red | The second link attached to the first link. |
| **Force Vector** | 🟢 Green | The control force applied to the cart. Length indicates magnitude, direction indicates sign. |
| **Track** | ⚫ Black | The horizontal rail the cart moves on. |

## Interpretation
*   **Goal**: The objective is to balance both the Blue and Red links in the upright position (vertical, above the cart).
*   **Force**: The Green line shows what the agent is doing. A line pointing right means positive force (pushing right), left means negative.
*   **Failures**: If the cart moves too far left/right (off-screen) or the pendulums swing too violently, the episode may terminate (depending on training logic).
