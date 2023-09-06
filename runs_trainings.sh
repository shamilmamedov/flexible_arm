DEVICE=0
SEED=0
SLEEPTIME=60


# TENSORBOARD
tmux new -d -s "TB" "tensorboard --logdir=logs --port=6006"

# BC
tmux new -d -s "BC_$SEED" "python -m tests.test_mpc_bc training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# DAGGER
tmux new -d -s "DAGGER_$SEED" "python -m tests.test_mpc_dagger training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# GAIL
tmux new -d -s "GAIL_$SEED" "python -m tests.test_mpc_gail training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# AIRL
tmux new -d -s "AIRL_$SEED" "python -m tests.test_mpc_airl training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# DENSITY
tmux new -d -s "DENSITY_$SEED" "python -m tests.test_mpc_density training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# SAC
tmux new -d -s "SAC_$SEED" "python -m tests.test_sac training.device=$DEVICE training.seed=$SEED"
sleep $SLEEPTIME

# PPO
tmux new -d -s "PPO_$SEED" "python -m tests.test_ppo training.device=$DEVICE training.seed=$SEED"



