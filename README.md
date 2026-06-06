# GEO RPOD Python Reference Simulation

Python reference plant, sensor simulation, Monte Carlo environment, and debug
sandbox for GEO rendezvous, proximity operations, and docking of a 50 kg
servicing deputy approaching a tumbling GEO chief.

The paired embedded C/SIL repository is:

```text
C:\Users\Venkat\OneDrive\Desktop\appex\Satellite_GNC
```

> Status: engineering prototype / reference plant. This is not flight-certified.

## Current Baseline

The current Python architecture uses a close-range dock-port/flash-lidar style
sensor handover for terminal docking. The earlier monocular pose-only terminal
failure mode was caused by estimator starvation: the translation loop reached
the port, camera features left the usable field of view, pose aged out, and the
attitude loop chased stale/ambiguous pose.

The current solution is:

- keep camera/PnP useful at longer terminal ranges,
- use the dock-port sensor at close range,
- gate pose freshness before spin-sync/dock-axis control,
- use soft capture at the 30 deg entry gate,
- hard capture only after strict 10 deg alignment and 5 s hold.

Latest reported Monte Carlo baseline:

```text
Trials total        : 20
Docking success     : 20 / 20 (100.0%)
Soft capture seen   : 20 / 20
Hard strict final   : 20 / 20
Soft certified      : 20 / 20
Capture timeouts    : 0 / 20
Mean total DV       : 1.491 m/s
95th percentile DV  : 1.932 m/s
Mean time to dock   : 1.776 hr
Final align mean    : 7.33 deg
Final align max     : 9.21 deg
```

Stress cases in that run:

```text
nominal          6 / 6
gyro_bias        6 / 6
high_pose_noise  3 / 3
camera_dropout   2 / 2
range_dropout    1 / 1
weak_thruster    2 / 2
```

## Architecture

Core layout:

```text
main.py                    single-run closed-loop simulation
monte_carlo.py             Monte Carlo campaign runner
sim_config.py              mission, sensor, guidance, and capture constants
chief_pose_estimator.py    chief attitude/omega estimator
control/                  RPOD, Lambert, spin-sync, guidance controllers
estimation/               MEKF, TH-EKF, terminal nav, port tracker
sensors/                  gyro, range, camera, dock-port/flash-lidar models
plant/                    deputy rigid body, finite body, contact dynamics
environment/              GEO orbit, CW dynamics, gravity gradient, SRP, sun, B-field
tools/                    telemetry and MC analysis scripts
tests/                    Python unit/regression tests
```

Mission flow:

```text
DETUMBLE
  -> SUN_ACQUISITION
  -> FINE_POINTING
  -> FORMATION_HOLD
  -> LAMBERT / PROX_OPS
  -> TERMINAL
  -> SOFT_CAPTURE
  -> DOCKING
```

The Python sim remains the truth plant and campaign environment. The C repo
ports the flight-software kernels and runs them against this plant for SIL/HIL.

## Current Scenario

```text
Chief orbit                 GEO, a = 42164 km, e = 0.0003
Deputy mass                 50 kg
Deputy thrust               1 N
Max acceleration            20 mm/s^2
Initial standoff            ~1000 m trailing
Inner ADCS step             0.01 s
RPOD outer-loop step        0.1 s
Terminal handoff            10 m
Port sensor range           10 m
Soft-capture gate           port < 0.30 m, vrel < 0.05 m/s, align < 30 deg
Hard-capture gate           port < 0.08 m, vrel < 0.010 m/s, align < 10 deg
Hard-capture hold           5 s
Dock cone half angle        15 deg
```

Docking geometry:

```text
DOCK_PORT_BODY = [0, 0, 0.5] m
DOCK_AXIS_BODY = [0, 0, 1]
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r Requirements.txt
```

Run one simulation:

```powershell
python main.py
```

Analyze the latest single-run telemetry:

```powershell
python tools\analyze_rpod_telemetry.py
```

Run Monte Carlo:

```powershell
python monte_carlo.py --trials 20 --workers 8 --seed 42
python tools\analyze_mc_results.py
```

Run tests:

```powershell
python -m pytest tests\ -v
```

## Telemetry To Watch

For terminal/debug work, the important telemetry is:

- truth alignment and estimated alignment
- truth-est alignment bias
- pose-estimator status counts: `ACCEPTED`, `REJECTED`, `COAST`, `NO_VISIBLE`
- visible point count and visible model indices
- stub/feature visibility
- pose age
- reprojection RMS
- port range and port relative velocity
- soft-capture entry, best, and final alignment

Healthy terminal behavior now looks like:

- terminal truth align mostly below 30 deg,
- close-range pose bias near single digits or lower,
- port sensor valid through final approach,
- soft capture enters below 30 deg,
- hard capture latches below 10 deg after the hold.

## Relationship To The C Port

Do not port `main.py` directly. It is the simulation harness, not flight code.

The C repo should port or mirror:

- guidance kernels,
- capture gates,
- estimators,
- mode manager,
- packet ABI,
- fault handling,
- actuator output limits,
- target/HIL communication.

The Python repo should remain responsible for:

- truth plant,
- sensors and stress cases,
- Monte Carlo campaigns,
- plots and diagnostics,
- reference behavior for C parity checks.

## Outputs

Generated outputs are local artifacts:

```text
rpod_telemetry.npz
rpod_telemetry_plots.png
monte_carlo_results.npz
monte_carlo_summary.txt
monte_carlo_plots.png
mc_results/
Figure_*.png
```

Preserve important campaign results in this README or a report, not by
accidentally committing raw generated artifacts.

## Near-Term Work

Python side:

- run larger MC sets after any guidance or sensor-model change,
- keep analysis scripts reporting pose freshness and port-sensor health,
- add named baseline reports for important campaigns,
- keep deterministic replay for failed or suspicious seeds.

C/HIL side lives in `Satellite_GNC`:

- target compiler build,
- MCU/flight-computer execution,
- real jitter/WCET measurement,
- real comms-link HIL with this Python plant.

## Git Hygiene

Commit source, tests, configuration, and documentation. Avoid committing:

```text
__pycache__/
.pytest_cache/
rpod_telemetry*.npz
monte_carlo_results*.npz
*_plots.png
Figure_*.png
mc_results/
```

## Author

Venkat Sainath

MSc Space Engineering
