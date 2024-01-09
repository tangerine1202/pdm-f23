skill := pick
num_envs := 2

train:
	@export MAGNUM_LOG=quiet; \
	export HABITAT_SIM_LOG=quiet; \
	set -x; \
      echo "Training $(skill) skill with $(num_envs) environments."; \
	python habitat_baselines/run.py \
	  --exp-config habitat_baselines/config/rearrange/ddppo_$(skill).yaml \
	  --run-type train \
	  TENSORBOARD_DIR ./$(skill)_tb/ \
	  CHECKPOINT_FOLDER ./$(skill)_checkpoints/ \
	  LOG_FILE ./$(skill)_train.log \
        NUM_ENVIRONMENTS $(num_envs); \
	echo "Training completed."
