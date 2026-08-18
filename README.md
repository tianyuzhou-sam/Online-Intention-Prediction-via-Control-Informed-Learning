<div align="center">

# Online Intention Prediction via Control-Informed Learning

**Tianyu Zhou · Zihao Liang · Zehui Lu · Shaoshuai Mou**
Purdue University

[**Paper**](https://arxiv.org/abs/2604.09303) ·
[**Code**](https://github.com/ZihaoLiang/Online-Intention-Prediction-via-Control-Informed-Learning)

[![arXiv](https://img.shields.io/badge/arXiv-2604.09303-b31b1b.svg)](https://arxiv.org/abs/2604.09303)

</div>

Watch a robot move and you can guess where it is going. This code does that
guessing online, one measurement at a time.

We treat the observed agent as an **optimal control system**. Its goal state
sits inside its own objective function as an unknown parameter, next to the
unknown parameters of its dynamics. Predicting the intention then means
estimating that parameter from the trajectory as it arrives.

Two parts do the work. A **gradient generator** differentiates Pontryagin's
Maximum Principle to get the exact derivative of the predicted trajectory with
respect to the parameters. An **extended Kalman filter** corrects the parameters
with each new measurement. A **shifting horizon** keeps the estimate current: at
each step the optimal control problem is re-solved from the state the agent was
in `MemoryTime` steps ago, so old data stops dragging the estimate along.

That last part is what lets the estimate follow an intention that changes. When
the agent switches target mid-flight, the goal error jumps and then falls again
within a few tens of steps, without any restart or re-training.

## How one step works

For every incoming measurement:

1. Solve the optimal control problem with the current parameter guess. This
   gives the trajectory the agent *would* follow if the guess were right.
2. Build the auxiliary control system from that solution and solve it. This
   gives `dξ/dθ`, the derivative of the trajectory with respect to the
   parameters.
3. Take the residual between the predicted state and the measured state.
4. Chain the two together into the filter's measurement matrix, `dL/dθ`.
5. Correct the parameters with one extended Kalman filter update.

The whole loop lives in `solve()` in each module, marked with those five
comments.

## The parameter vector

For the quadrotor the tunable parameter `theta` has 24 entries:

| Entries | Meaning |
|---|---|
| 0–5 | dynamics: `Jx, Jy, Jz, mass, l, c` |
| 6–10 | cost weights: `wthrust, wr, wv, wq, ww` |
| 11–23 | **the intention**: goal position (3), goal velocity (3), goal attitude (4), goal angular rate (3) |

Only the thrust term is a running cost. The goal terms are the terminal cost, so
the demonstrator is a minimum-effort mover heading for a terminal state. The
filter is told to trust entries 11–13 least — the scripts inflate those diagonal
entries of `P` — because the goal position is what we are actually after.

## Installation

The submodule carries the optimal control solver and the gradient machinery, so
clone recursively:

```bash
git clone --recursive https://github.com/ZihaoLiang/Online-Intention-Prediction-via-Control-Informed-Learning.git
cd Online-Intention-Prediction-via-Control-Informed-Learning
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Then the dependencies. Tested on Python 3.10 with NumPy 1.26, SciPy 1.15,
Matplotlib 3.10, CasADi 3.7 and transforms3d 0.4:

```bash
conda create -n oip python=3.10
conda activate oip
pip install numpy scipy matplotlib casadi transforms3d
pip install pandas   # only for the hardware scripts in experiment/
```

## Quick start

**Run every script from the repository root.** Each one builds its import paths
from the working directory, so launching from inside `example/` will not find
`src/`.

```bash
python example/OCIL_prediction.py
```

A quadrotor flies to `[2, 10, 1]`, switches target to `[10, 8, 1]` at step 50,
then to `[0, 5, 1]` at step 80. The printed `L goal` is the squared error of the
whole 13-entry goal vector. A typical run starts at 105, falls to about 6 by
step 49, jumps to 74 at the switch, falls to 3 by step 79, jumps again, and is
back near 14 by the end. About 0.14 s per step on one CPU core.

## Simulation examples

Every script sets up a quadrotor, hands the true system to a simulated
demonstrator, and asks the learner to recover the goal. `example/`:

| Script | Module | What it shows |
|---|---|---|
| `OCIL_prediction.py` | `src/OCIL.py` | Full-horizon prediction, two hand-set target switches |
| `switch_target.py` | `src/ImitationLearningMPC.py` | The same target schedule with the shifting horizon, about four times faster per step |
| `track_target.py` | `src/ImitationLearningMPC.py` | A target that drifts 0.1 m per step, so the intention never settles |
| `OCIL_prediction_multi.py` | `src/OCILSwitch.py` | 100 trials, random new target every `window` steps, full horizon |
| `switch_target_multi.py` | `src/ImitationLearningSwitch.py` | The same 100 trials with the shifting horizon. This is the pair the figures compare |
| `uav_prediction.py` | `src/ImitationLearning.py` | Fixed target, no switching. The base case |
| `nn_uav_prediction.py` | `src/ImitationLearning.py` | Unknown dynamics as a neural network, learned alongside the goal |
| `generateDemo*.py` | `src/generateTraj.py` | Write a demonstration trajectory to `.mat` for the hardware scripts |

The four modules differ only in how much of the trajectory they use:

- `OCIL.py` and `OCILSwitch.py` solve the whole horizon from the fixed initial
  state and read the sensitivity at the current index.
- `ImitationLearningMPC.py` and `ImitationLearningSwitch.py` re-solve from the
  state `MemoryTime` steps back over a shrinking horizon, and always read the
  sensitivity at index `MemoryTime`. This is the shifting horizon.
- The `*Switch` pair picks a new random target every `window` steps, for
  Monte-Carlo runs. The other two take a target schedule you write out by hand
  through `switch_target(switch_time, switch_goal)`.

`costnn_prediction.py` calls `initNeuralCost`, which is not in `src/Env.py`, so
it does not run as it stands.

## Hardware experiments

`experiment/` holds the motion-capture logs and a separate copy of the learner
that reads them. The goal is estimated from the recorded trajectory alone.

```bash
python experiment/dog_prediction.py     # Unitree quadruped, 2D goal
python experiment/quad_prediction.py    # Crazyflie-style quadrotor, 3D goal
```

The dog run ends with the goal estimated at about `[2.70, -0.81]` against a true
goal of `[2.73, -0.83]`. The quadrotor run ends at about `[2.19, 0.50, -0.07]`
against `[2, 0, 0.6]`.

| File | What it is |
|---|---|
| `dog_data.csv`, `quad_data.csv` | Qualisys motion-capture logs, cropped from 9 s and 11 s |
| `quad1..7.csv`, `data2..5.csv` | Further raw takes |
| `prediction{dog,quad}.mat` | Saved prediction results, read by the visualization scripts |
| `run_dog.py` | Plays a saved control trajectory on a Unitree dog over UDP |
| `run_mocap_qualisys_for_dog.py` | Streams poses from the Qualisys server |
| `visualization{dog,quad}.py` | Draw the figures from the saved results |

`run_dog.py` and `run_mocap_qualisys_for_dog.py` need the robot SDK and a live
motion-capture server, so they only run in the lab.

## Code layout

```
src/
  OCIL.py                     full-horizon prediction  (+ SysID, PolicyTuning)
  OCILSwitch.py               full horizon, random target switches
  ImitationLearningMPC.py     shifting horizon
  ImitationLearningSwitch.py  shifting horizon, random target switches
  ImitationLearning.py        fixed target; ImitationLearningNN for neural dynamics
  EKF.py                      the predict and update steps, 20 lines
  Env.py                      the quadrotor: dynamics, cost, neural dynamics, animation
  generateTraj.py             write a demonstration to .mat
  compare_*.py, plot_*.py     the figures
externals/
  Pontryagin-Differentiable-Programming/
                              optimal control solver, auxiliary control system,
                              LQR solver used for the gradient
example/                      the simulation scripts above
experiment/                   hardware data, a separate learner, hardware drivers
```

## Reproducing the figures

Set `saveFlag = True` in a script and every run writes
`results/result_<project>.mat`, carrying the loss history, the goal error, the
whole parameter history, the predicted and demonstrated trajectories, and the
per-step timings.

The plotting scripts read runs that were sorted into subfolders by hand, and
those folders are not in the repository. You have to produce them first:

| Script | Reads | Which runs |
|---|---|---|
| `src/compare_normal.py`, `compare_uniform.py` | `results/noise_0/`, `normal_02/`, `normal_05/`, `uniform_01/`, `uniform_05/`, `uniform_1/` | 100 trials per noise level |
| `src/compare_multi_switch.py`, `compare_loss.py` | `results/10/` … `results/60/` and `results/10_OCIL/` … `results/60_OCIL/` | shifting horizon against full horizon, by switch interval |
| `src/compare_nn.py` | `results/nn/`, `nn_1/`, `nn_3/` | neural dynamics, by network size |
| `src/plot_Tm.py` | `results/Tm_5.mat`, `Tm_10.mat`, `Tm_20.mat` | one run per memory length |
| `src/plot_track.py` | `results/dt_1.mat`, `dt_2.mat`, `dt_5.mat`, `dt_10.mat` | one run per target speed |
| `src/plot_switch_target.py`, `src/runtime.py` | `results/switch_target.mat`, `results/noise_0/` | the switching figure and the timing table |

## Notes

- **The plots block.** Runs end with a blocking Matplotlib window. For
  unattended runs set a non-interactive backend: `MPLBACKEND=Agg python ...`

- **The filter needs tuning per case.** The covariances `P`, `Q` and `R` at the
  bottom of each script decide how fast the estimate moves and whether it
  survives noise. Each script keeps the working settings for other noise levels
  in a commented block at the end — those are the ones the paper used, not
  leftovers.

- **`MemoryTime` is the trade-off.** Short memory follows a changing intention
  quickly but is noisy; long memory is smooth but lags behind a switch.
  `src/plot_Tm.py` draws that comparison.

- **Cost per step is high.** Every measurement re-solves the full optimal
  control problem and the full auxiliary system, then uses one time index of the
  result. That is what buys an exact gradient. The shifting horizon keeps it
  affordable because the horizon shrinks as the run goes on.

## Citation

```bibtex
@article{zhou2026online,
  title   = {Online Intention Prediction via Control-Informed Learning},
  author  = {Zhou, Tianyu and Liang, Zihao and Lu, Zehui and Mou, Shaoshuai},
  journal = {arXiv preprint arXiv:2604.09303},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.09303}
}
```

## Acknowledgements

The optimal control solver, the auxiliary control system used to generate
gradients, and the simulated environments come from
[Pontryagin Differentiable Programming](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming)
by Wanxin Jin and colleagues. The online estimator follows
[Online Control-Informed Learning](https://github.com/ZihaoLiang/OCIL-Online-Control-Informed-Learning).
