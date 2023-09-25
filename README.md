# Safe Imitation Learning of Nonlinear Model Predictive Control for Flexible Robots
## Method

<img src="https://github.com/shamilmamedov/flexible_arm/assets/59015432/cfe07419-77ac-4b5c-a711-15e4d1eba2cd" alt="Image Description" width="800">

## Installation of acados:
Installation of acados according to the following instructions:
https://docs.acados.org/python_interface/index.html

## Imitation Library Fork (submodule):
Current (21 August 2023) version on imitation library does not yet support
Gymnasium. So we are using our own fork of it with necessary modifications. 

After cloning this repo:

- ```git submodule init```
- ```git submodule update```
- ```cd imitation```
- ```pip install -e .```

## Hyperparameters of the IL, RL and IRL algorithms

| Hyper-parameter                            | Value        |
|--------------------------------------------|--------------|
| COMMON: Learning Rate                      | 0.0003       |
| COMMON: Number of Expert Demos             | 100          |
| COMMON: Number of Training Steps           | 2,000,000    |
| PPO: Net. Arch.                            | pi:[256, 256] vf:[256, 256] |
| PPO: Batch Size                            | 64           |
| SAC: Net. Arch.                            | pi:[256, 256] qf:[256, 256] |
| SAC: Batch Size                            | 256          |
| BC: Net. Arch.                             | pi:[32, 32] qf:[32, 32]      |
| BC: Batch Size                             | 32           |
| DAgger: Online Episodes                    | 500          |
| Density: Kernel type                       | Gaussian     |
| Density: Kernel bandwidth                  | 0.5          |
| Density: Net. Arch.                        | pi:[256, 256] qf:[256, 256] |
| GAIL: Reward Net Arch.                     | [32, 32]     |
| GAIL: Policy Net Arch.                     | pi:[256, 256] qf:[256, 256] |
| GAIL: Policy Replay Buffer Capacity        | 512          |
| GAIL: Batch Size                           | 128          |
| AIRL: Reward Net Arch.                     | [32, 32]     |
| AIRL: Policy Net Arch.                     | pi:[256, 256] qf:[256, 256] |
| AIRL: Batch Size                           | 128          |
| AIRL: Policy Replay Buffer Capacity        | 512          |


## NMPC parameters
| Parameter                                  | Value                          |
|--------------------------------------------|--------------------------------|
| Hessian Approximation                      | Gauss-Newton                    |
| SQP type                                   | real-time iterations            |
| $\Delta t$, $N$, $n_\mathrm{seg}$          | $5$ ms, 125, 3                  |
| $Q$ weights $w_{q_a}$, $\dot w_{q_a}$, $w_{q_p}$, $\dot{w}_{q_p}$ | $0.01 \; 0.1 \; 0.01 \; 10$ |
| $P_N$                                      | diag($[1,1,1,0,0,0])\cdot 10^4$ |
| $P$                                        | diag($[1,1,1,0,0,0])\cdot 2\cdot10^3$ |
| $R$                                        | diag($[1,10,10]$)               |
| $S$, $s$                                   | diag($[1,1,1]\cdot 10^6$), $[1,1,1]^\top\cdot 10^4$ |
| $\delta_\mathrm{ee}, \delta_\mathrm{elb}$ , $\delta_\mathrm{x}$ | $0.01\mathrm{m}, \;0.005\mathrm{m}$, \; $0\cdot 1_{n_x}$ |
| $\overline{\dot{q_a}}=-\underline{\dot{q_a}}$           | $[2.5, 3.5, 3.5]^\top\;s^{-1}$  |
| $\overline{u}=-\underline{u}$              | $[20,10,10]^\top$ Nm            |


## Safety Filter parameters
| Parameter                                  | Value                        |
|--------------------------------------------|------------------------------|
| $\Delta t_\mathrm{SF}$, $N_\mathrm{SF}$, $n_\mathrm{seg}$ | $10$ ms, $25$, $1$ |
| $\bar{R}$                                  | diag($[1,1,1]$)               |
| ${R}_\mathrm{SF}$                          | diag($[1,1,1]$) $\cdot 10^{-5}$  |

