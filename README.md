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

