DEVICE=0
SEED=0


# TENSORBOARD
tmux new -d -s "TB" "tensorboard --logdir=logs --port=6006"

# BC
tmux new -d -s "BC_$SEED" "python -m tests.test_mpc_bc device=$DEVICE seed=$SEED"

# DAGGER
tmux new -d -s "DAGGER_$SEED" "python -m tests.test_mpc_dagger device=$DEVICE seed=$SEED"

# GAIL
tmux new -d -s "GAIL_$SEED" "python -m tests.test_mpc_gail device=$DEVICE seed=$SEED"

# AIRL
tmux new -d -s "AIRL_$SEED" "python -m tests.test_mpc_airl device=$DEVICE seed=$SEED"

# DENSITY
tmux new -d -s "DENSITY_$SEED" "python -m tests.test_mpc_density device=$DEVICE seed=$SEED"

# SAC
tmux new -d -s "SAC_$SEED" "python -m tests.test_sac device=$DEVICE seed=$SEED"

# PPO
tmux new -d -s "PPO_$SEED" "python -m tests.test_ppo device=$DEVICE seed=$SEED"



