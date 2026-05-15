# GEO RPOD Flight Simulation

Python high-fidelity simulation for a GEO rendezvous, proximity operations, and docking (RPOD) mission using a 50 kg deputy spacecraft servicing a tumbling GEO chief.

The model is built as an end-to-end GNC simulation: orbital environment, relative dynamics, sensors, attitude estimation, relative navigation, ADCS, Lambert transfer, terminal docking-port targeting, and Monte Carlo robustness analysis.

This repository is the Python reference model. The embedded C flight-software port and SIL harness live in the companion `Satellite_GNC` repository.

## Current Verification Status

Latest Monte Carlo run:

- Trials: `300`
- Docked: `300 / 300`
- Docking success: `100.0%`
- Mean total delta-V: `7.136 m/s`
- Median total delta-V: `6.020 m/s`
- 95th percentile delta-V: `14.713 m/s`
- Mean time to dock: `2.388 hr`
- 95th percentile time to dock: `3.800 hr`
- Mean propellant: `165.0 g` at `Isp = 220 s`
- 95th percentile propellant: `339.7 g`

The latest summary is saved in:

```text
monte_carlo_summary.txt
```

The latest Monte Carlo binary results are saved in:

```text
monte_carlo_results.npz
```

The latest plot bundle is:

```text
monte_carlo_plots.png
```

## Key Figures

### Monte Carlo Performance

![GEO RPOD Monte Carlo](monte_carlo_plots.png)

This figure shows the total delta-V distribution, terminal/proximity delta-V split, and sensitivity of total delta-V to chief tumble rate and mission duration.

### ADCS / Mode Timeline

![ADCS overview](Figure_1.png)

This figure shows attitude rate convergence, disturbance torques, eclipse function, reaction-wheel momentum, MEKF pointing error, and FSW mode progression.

### RPOD Trajectory

![RPOD trajectory](Figure_2.png)

This figure shows LVLH relative motion, truth-vs-EKF position channels, log-range closure, cumulative delta-V, and the final close-approach trajectory into the docking port.

## Mission Scenario

The reference scenario models a 50 kg deputy spacecraft operating near GEO.

| Item | Value |
|---|---:|
| Chief orbit | GEO, `a = 42164 km`, `e = 0.0003`, `i = 0.8 deg` |
| Chief longitude | `342.0 deg E` |
| Deputy mass | `50 kg` |
| Deputy thrust | `1 N` |
| Deputy max acceleration | `20 mm/s^2` |
| Initial standoff | `1000 m` trailing |
| Inner ADCS step | `0.01 s` |
| RPOD outer-loop step | `0.1 s` |
| Docking capture radius | `0.30 m` |
| Docking relative-speed gate | `0.05 m/s` |
| Terminal takeover | `5 m` main-loop override |
| Formation-hold EKF settle | `300 s` |

## Architecture

```text
main.py
  |
  +-- environment/
  |     GEO orbit, CW dynamics, SRP, drag, magnetic field,
  |     gravity gradient, sun model, chief tumble
  |
  +-- plant/
  |     deputy spacecraft rigid-body attitude dynamics
  |
  +-- sensors/
  |     gyro, magnetometer, sun sensor, star tracker,
  |     ranging/bearing sensor, camera/PnP-style port sensor
  |
  +-- estimation/
  |     QUEST, MEKF, TH-EKF
  |
  +-- control/
  |     Lambert solver, RPOD phase controller, ADCS controller
  |
  +-- fsw/
        high-level mode manager
```

## Flight Sequence

The nominal mission progresses through:

```text
DETUMBLE
  -> SUN_ACQUISITION
  -> FINE_POINTING
  -> FORMATION_HOLD
  -> LAMBERT
  -> PROX_OPS
  -> TERMINAL
  -> DOCKING
```

Major behaviors:

- B-dot detumbling reduces initial body rates.
- QUEST seeds the attitude estimate.
- MEKF maintains fine pointing using gyro/vector measurements.
- TH-EKF estimates relative position and velocity.
- Lambert guidance moves from standoff to close approach.
- PROX_OPS uses continuous sqrt-law closure.
- TERMINAL targets the docking port, not just the chief center of mass.
- Docking is confirmed using port range and relative velocity.

## Main Components

### Attitude Dynamics and ADCS

The deputy attitude model includes rigid-body rotational dynamics, reaction wheels, magnetorquers, environmental disturbances, and closed-loop pointing control.

ADCS outputs tracked in the plots include:

- angular rate,
- wheel momentum,
- disturbance torques,
- MEKF pointing error,
- mode-manager state.

### Relative Dynamics

Relative motion is propagated in LVLH using CW-style dynamics around a GEO chief. The simulation includes mission-relevant GEO perturbation terms such as differential solar radiation pressure.

### Navigation

Navigation is split into:

- MEKF for attitude and gyro-bias estimation,
- TH-EKF for relative orbit estimation,
- camera/PnP-style chief pose and docking-port updates in close approach.

Terminal logic uses direct close-range port measurements when available to avoid late-stage estimator lag.

### RPOD Guidance

The RPOD controller includes:

- formation hold,
- Lambert transfer,
- lost-target handling,
- PROX_OPS approach,
- TERMINAL docking-port closure,
- docking detection.

The terminal controller uses a speed-limited range law and a docking-port target derived from chief pose.

### Chief Attitude and Docking Port

The chief is modeled as a tumbling target with a body-fixed docking port:

```text
DOCK_PORT_BODY = [0, 0, 0.5] m
DOCK_AXIS_BODY = [0, 0, 1]
```

The simulation estimates and tracks the port in LVLH during terminal approach. This is one of the key differences between a simple translational rendezvous simulation and a 6-DOF servicing-relevant RPOD model.

## Monte Carlo

Run the Monte Carlo with:

```bat
python monte_carlo.py --trials 300 --workers 8
```

Outputs:

```text
monte_carlo_results.npz
monte_carlo_summary.txt
monte_carlo_plots.png
```

Latest statistical result:

| Metric | Mean | Std | 5th % | Median | 95th % |
|---|---:|---:|---:|---:|---:|
| Total delta-V (m/s) | 7.136 | 4.061 | 3.727 | 6.020 | 14.713 |
| PROX_OPS delta-V (m/s) | 0.644 | 0.055 | 0.610 | 0.632 | 0.713 |
| TERMINAL delta-V (m/s) | 6.492 | 4.062 | 3.092 | 5.394 | 14.064 |
| Time to dock (hr) | 2.388 | 1.850 | 1.556 | 1.909 | 3.800 |
| Chief tumble (deg/s) | 0.148 | 0.060 | 0.059 | 0.148 | 0.236 |

Notes:

- Most of the delta-V is consumed in terminal docking-port tracking.
- PROX_OPS delta-V is tightly clustered around `0.6-0.7 m/s`.
- The long right tail in total delta-V is driven by difficult terminal geometry and longer docking timelines.
- The latest run shows weak correlation between total delta-V and chief tumble rate (`-0.088`) and modest positive correlation with time-to-dock (`+0.239`).

## Single-Run Simulation

Run the nominal mission:

```bat
python main.py
```

Expected outputs:

- console mission timeline,
- docking confirmation if successful,
- ADCS summary figure,
- RPOD trajectory figure.

## Requirements

Install Python dependencies from:

```bat
pip install -r Requirements.txt
```

Core dependencies include:

- `numpy`
- `scipy`
- `matplotlib`

The model is developed and exercised on Windows, but the code is pure Python apart from local path assumptions in some scripts.

## Validation Philosophy

The Python model is used as the algorithmic reference for:

- mission feasibility,
- guidance-law tuning,
- estimator behavior,
- docking-port terminal approach,
- Monte Carlo performance,
- embedded C port comparison.

The companion C implementation should be treated as the flight-software candidate. This Python repository remains the higher-fidelity reference and analysis environment.

## Current Limitations

This is an engineering simulation, not a certified flight dynamics tool.

Known limitations:

- Contact dynamics are simplified; docking confirmation stops the run rather than simulating hard/soft capture mechanics.
- Docking-port pose is modeled through simulated chief attitude and pose estimation, not a full optical image-processing stack.
- GEO environmental models are suitable for GNC trade studies, not final mission operations.
- Thermal, power, communications, plume impingement, and flexible-body dynamics are out of scope.
- The Monte Carlo randomizes important RPOD parameters, but it is not yet a full mission assurance campaign.
- FDIR is represented at the mode/guidance level, but a formal fault tree, FMECA matrix, and fault-injection campaign are still future work.

## Industry-Grade Next Steps

Before presenting this as a flight-ready GNC stack, the next review items are:

1. Add a formal FDIR/FMECA table with hazard severity, detection logic, response, and verification test for each fault.
2. Add fault-injection Monte Carlo cases: sensor dropout, stale packets, camera spikes, gyro bias jumps, port loss, actuator saturation, and missed guidance updates.
3. Add multi-sample docking confirmation and explicit post-contact state handling.
4. Add approach-axis and attitude-alignment gates for docking.
5. Replace local path assumptions with a config file or environment variables.
6. Version the Monte Carlo configuration with each results file.
7. Cross-check Python results against the C SIL after every major guidance or estimator change.

## Relationship to Embedded C Repo

The Python simulation is the reference model. The C flight-software repository implements the embedded/SIL side:

```text
Satellite_GNC/
  src_c/
  sim_python/
  tests/
```

The C repo currently mirrors the key RPOD, TH-EKF, MEKF, ADCS, and mode-manager logic and is verified through C unit tests plus Python closed-loop SIL.

## Notes for Version Control

Recommended to commit:

- source code,
- requirements,
- README,
- small summary artifacts such as `monte_carlo_summary.txt`,
- selected result figures when useful for review.

Recommended to ignore or avoid committing by default:

- large telemetry CSVs,
- `__pycache__/`,
- temporary plots,
- large raw Monte Carlo archives unless they are part of a tagged result release.

## Author

Venkat Sainath  
MSc Space Engineering
