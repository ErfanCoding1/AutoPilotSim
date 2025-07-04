# Autonomous Vehicle Sensor Integration and Path Tracking

This project is a practice-based implementation of autonomous vehicle control using multiple sensors and a PID controller. It consists of three main tasks that build upon each other to achieve a closed-loop autonomous driving behavior.

## Task 1: Sensor Integration and Basic Control

In this step, I integrated RGB camera, IMU, and LIDAR sensors into a simulated vehicle. The vehicle was controlled to follow a predefined path using a PID controller. Sensor data (images, IMU readings, and point clouds) were stored locally for further analysis and debugging purposes.

## Task 2: PID Tuning

I focused on tuning the PID gains for both **longitudinal** (speed) and **lateral** (steering) control. Several important criteria were considered for effective tuning, including:

- **Stability:** Ensuring that the vehicle does not oscillate.
- **Responsiveness:** Achieving fast response to path deviations.
- **Accuracy:** Minimizing cross track error (CTE).

I explored how each individual gain (P, I, D) affects system behavior:

- **Proportional (P):** Controls the responsiveness to error but can cause overshoot.
- **Integral (I):** Eliminates steady-state error but may introduce lag and instability.
- **Derivative (D):** Dampens oscillations and improves stability.

Various PID gain configurations were tested. I plotted and compared:

- Control actions
- Cross track error (CTE)
- Vehicle position and speed
- Desired vs actual trajectory

Based on the performance comparison, I selected a final set of PID gains that gave the best balance between accuracy and smoothness.

## Task 3: Infinite Looped Trajectory

A closed-loop trajectory was created, and the vehicle was successfully controlled to follow it indefinitely. The autonomous system was able to continuously track the loop without divergence or instability.

---


## Requirements

- Python 3.8+
- CARLA Simulator
- NumPy, Matplotlib, Pandas

Install requirements using:

```bash
pip install -r requirements.txt



