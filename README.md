# GEO RPOD Python Reference Simulation

Python reference plant, sensor simulation, Monte Carlo environment, and debug
sandbox for GEO rendezvous, proximity operations, and docking of a 50 kg
servicing deputy grappling the engine bell of a tumbling GEO chief (IS-1002
class comsat, ~50 kg jetpack servicer).

The paired embedded C/SIL repository is:

```text
C:\Users\Venkat\OneDrive\Desktop\appex\Satellite_GNC
```

> Status: engineering prototype / reference plant. Not flight-certified.

## Current Baseline

**Scenario: non-cooperative uncontrolled tumbler, engine-bell grapple.**

The deputy approaches from the -Z (nozzle) side of the chief, aligns with the
nozzle axis, and captures the LAE engine bell using a flash-lidar / RANSAC
nozzle estimator. There is no cooperative docking port. The chief is tumbling
at 0.5–2 deg/s and does not respond to commands.

Pipeline:

- SURVEY phase: lidar point cloud + RANSAC circle fitter estimates nozzle centre
  and radius, builds confidence/stability score
- TERMINAL gate: opens when nozzle estimator confidence and stability thresholds
  are met
- bell capture: proximity trigger replaces geometric dock-port gates; a 5-second
  BELL_HARD_HOLD timer runs before DOCKING is declared
- false-positive rejection: 6-axis probe checks the fitted circle against the
  full body mesh before propagating to the estimator

Latest Monte Carlo baseline (20 trials, seed 42):

```text
Trials total        : 20
Docking success     : 20 / 20  (100.0%)
Soft capture seen   : 20 / 20
Capture timeouts    :  0 / 20
Mean total DV       : 3.028 m/s
95th-pct DV         : 3.654 m/s
Mean time to dock   : 1.603 hr
95th-pct time       : 1.866 hr
Nozzle conf (mean)  : 1.000  (estimator always converges before TERMINAL)
```

Stress-case breakdown:

```text
nominal              5 / 5
gyro_bias            1 / 1
high_pose_noise      1 / 1
camera_dropout       1 / 1
slow_detumble        1 / 1
weak_thruster        2 / 2
lidar_occlusion      5 / 5   (90-s lidar dropout during SURVEY)
nozzle_oversize      2 / 2   (nozzle radius +15%)
nozzle_undersize     2 / 2   (nozzle radius -15%)
```

Note: `range_dropout` was not sampled in this 20-trial run (low weight, draws
randomly). Re-run with `--trials 50` to populate sparse cases.

## Architecture

Core layout:

```text
main.py                    single-run closed-loop simulation (argparse CLI)
monte_carlo.py             Monte Carlo campaign runner
sim_config.py              mission, sensor, guidance, and capture constants
render/
  chief_renderer.py        software rasteriser + IS-1002 mesh (138 triangles)
                           including Ka-band antenna bracket and array hinges
chief_pose_estimator.py    chief attitude/omega estimator
estimation/
  nozzle_estimator.py      flash-lidar RANSAC circle → nozzle centre + radius
control/                   RPOD, Lambert, spin-sync, guidance controllers
estimation/                MEKF, TH-EKF, terminal nav
sensors/
  lidar_pointcloud_sensor.py  flash lidar (Möller–Trumbore ray cast, geometry override)
  camera.py                   monocular camera model
  dock_port_sensor.py         cooperative dock-port sensor (inactive in uncoop mode)
plant/                     deputy rigid body, finite body, contact dynamics
environment/               GEO orbit, CW dynamics, gravity gradient, SRP, sun, B-field
tools/
  analyze_rpod_telemetry.py   single-run NPZ analyser (mode-aware: coop vs uncoop)
  analyze_mc_results.py       Monte Carlo PKL analyser
  update_report_current_state.py  auto-update Word report
tests/                     Python unit/regression tests
```

Mission flow (non-cooperative):

```text
DETUMBLE
  -> SUN_ACQUISITION
  -> FINE_POINTING
  -> FORMATION_HOLD
  -> LAMBERT / PROX_OPS
  -> SURVEY          (lidar active, nozzle estimator building confidence)
  -> TERMINAL        (gated on estimator confidence + stability thresholds)
  -> SOFT_CAPTURE    (proximity contact, bell geometry check)
  -> DOCKING         (5 s BELL_HARD_HOLD, then declared)
```

The Python sim is the truth plant and campaign environment. The C repo ports
flight-software kernels and runs against this plant for SIL/HIL.

## Current Scenario

```text
Chief orbit                 GEO, a = 42164 km, e = 0.0003
Chief attitude              uncontrolled tumble 0.5–2 deg/s (random axis)
Deputy mass                 50 kg
Deputy thrust               1 N
Max acceleration            20 mm/s^2
Initial standoff            ~1000 m trailing
Inner ADCS step             0.01 s
RPOD outer-loop step        0.1 s
Approach axis               chief -Z (nozzle face)

LAE nozzle geometry
  Base radius               0.185 m
  Exit radius               0.240 m
  Nozzle length             0.420 m

Soft-capture trigger        proximity to bell + geometry check
Hard-capture gate           bell contact + 5 s hold (BELL_HARD_HOLD_S)
SURVEY entry range          ≤ 50 m
TERMINAL entry              nozzle_conf ≥ 0.70, stability ≥ 0.60,
                            drift ≤ 0.10 m/s, FP score ≥ 0.80
```

Lidar sensor:

```text
Rays per shot               60
Range noise (1-sigma)       0.02 m  (nominal); scaled per stress profile
RANSAC search band          R ∈ [0.10, 0.45] m
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run one simulation (headless, non-cooperative default):

```powershell
python main.py --no-plot
```

Run cooperative mode (dock-port target, legacy):

```powershell
python main.py --no-plot --cooperative
```

Reproducible single run:

```powershell
python main.py --no-plot --seed 7
```

Analyze the latest single-run telemetry:

```powershell
python tools\analyze_rpod_telemetry.py          # saves PNG
python tools\analyze_rpod_telemetry.py --show   # opens window
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

## CLI Flags (main.py)

```text
--no-plot        run headless (Agg backend, no plt.show)
--cooperative    switch to cooperative dock-port mode (default: non-cooperative)
--seed N         RNG seed for repeatable runs (default: 42)
--out-dir PATH   directory for output files (default: cwd)
```

## Telemetry To Watch

For terminal/debug work in non-cooperative mode:

- **nozzle estimator**: confidence, stability, drift, FP score at TERMINAL entry
- **SURVEY duration**: time from SURVEY entry to TERMINAL gate open
- flash lidar health: point cloud density, dropout windows
- truth nozzle-axis alignment and estimated alignment
- truth-est alignment bias
- pose-estimator status counts: `ACCEPTED`, `REJECTED`, `COAST`, `NO_VISIBLE`
- pose age, reprojection RMS, PCA condition number
- bell capture trigger: proximity check + `uncoop_override_fired`

Note on port metrics: `rn_port_range` and related cooperative-port metrics are
logged throughout but are **not** the capture criterion in non-cooperative mode.
`analyze_rpod_telemetry.py` labels them accordingly when `uncooperative_mode`
is set in the NPZ.

Healthy non-cooperative terminal behavior:

- nozzle_conf ≥ 0.70 before TERMINAL entry,
- nozzle estimator still stable at contact (conf = 1.0 in all 20 baseline trials),
- deputy +Z vs nozzle axis alignment below 30 deg through approach,
- bell proximity trigger fires within a few seconds of contact.

## Relationship To The C Port

Do not port `main.py` directly. It is the simulation harness, not flight code.

The C repo should port or mirror:

- nozzle estimator (RANSAC circle fitter + confidence logic),
- SURVEY/TERMINAL mode manager and gate thresholds,
- bell-capture proximity check,
- MEKF, TH-EKF, terminal nav,
- guidance kernels and spin-sync,
- packet ABI, fault handling, actuator limits.

The Python repo remains responsible for:

- truth plant,
- mesh renderer and lidar ray caster,
- sensors and stress cases,
- Monte Carlo campaigns,
- plots and diagnostics,
- reference behavior for C parity checks.

## Outputs

Generated outputs are local artifacts (gitignored):

```text
rpod_telemetry.npz
rpod_telemetry_plots.png
mc_results.pkl
ca_results.png
monte_carlo_results/
Figure_*.png
```

Preserve important campaign results in this README or a report. Do not commit
raw generated artifacts.

## Near-Term Work

Python side:

- run larger MC sets (50–100 trials) to populate sparse stress cases,
- add named baseline reports for each major architecture change,
- keep deterministic replay for failed or suspicious seeds (`--seed N`),
- pose CNN integration (ENABLE_CNN_POSE_ESTIMATOR flag, currently False).

C/HIL side lives in `Satellite_GNC`:

- port nozzle estimator and SURVEY/TERMINAL mode manager,
- target compiler build and WCET measurement,
- real comms-link HIL with this Python plant.

## Git Hygiene

Commit source, tests, configuration, and documentation. Avoid committing:

```text
__pycache__/
.pytest_cache/
rpod_telemetry*.npz
mc_results.pkl
monte_carlo_results/
*_plots.png
Figure_*.png
```

## Author

Venkat Sainath — MSc Space Engineering
