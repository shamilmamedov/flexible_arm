# Reinforcement Learning for flexible arm robots

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
|-------------------------------------------|--------------|
| COMMON: Learning Rate                     | 0.0003       |
| COMMON: Number of Expert Demos            | 100          |
| COMMON: Number of Training Steps          | 2,000,000    |
| PPO: Net. Arch.                           | pi:[256, 256] vf:[256, 256] |
| PPO: Batch Size                           | 64           |
| SAC: Net. Arch.                           | pi:[256, 256] qf:[256, 256] |
| SAC: Batch Size                           | 256          |
| BC: Net. Arch.                            | pi:[32, 32] qf:[32, 32]      |
| BC: Batch Size                            | 32           |
| DAgger: Online Episodes                   | 500          |
| Density: Kernel type                      | Gaussian     |
| Density: Kernel bandwidth                 | 0.5          |
| Density: Net. Arch.                      | pi:[256, 256] qf:[256, 256] |
| GAIL: Reward Net Arch.                    | [32, 32]     |
| GAIL: Policy Net Arch.                    | pi:[256, 256] qf:[256, 256] |
| GAIL: Policy Replay Buffer Capacity       | 512          |
| GAIL: Batch Size                          | 128          |
| AIRL: Reward Net Arch.                    | [32, 32]     |
| AIRL: Policy Net Arch.                    | pi:[256, 256] qf:[256, 256] |
| AIRL: Batch Size                          | 128          |
| AIRL: Policy Replay Buffer Capacity       | 512          |
