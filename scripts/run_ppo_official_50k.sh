#!/usr/bin/env bash
# Run official ManiSkill PPO on PickCube-v1 and PegInsertionSide-v1, seeds 0/1/2,
# 50k env-steps each, 2 workers in parallel. physx_cpu because GPU sim is broken
# on this box (WSL2 + GTX 1650).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deeplearning

export MS_SIM_BACKEND=physx_cpu
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export SAPIEN_DISABLE_RAY_TRACING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT=outputs/ppo_official_50k
mkdir -p "$OUT/logs"

run_one() {
  local task=$1 seed=$2
  local tag="${task}_s${seed}"
  local log="$OUT/logs/${tag}.log"
  echo "[$(date -Is)] START $tag  -> $log"
  python scripts/ppo_official.py \
    --env_id="$task" --seed="$seed" \
    --num_envs=1 --num_eval_envs=1 \
    --num_steps=1024 --num_eval_steps=500 \
    --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=50000 --eval_freq=1 \
    --no-capture_video --no-save_model \
    --exp_name="ppo_off_50k_${tag}" \
    > "$log" 2>&1
  echo "[$(date -Is)] DONE  $tag  exit=$?"
}
export -f run_one
export OUT

# 2 workers via xargs -P2
printf "PickCube-v1 0\nPickCube-v1 1\nPickCube-v1 2\nPegInsertionSide-v1 0\nPegInsertionSide-v1 1\nPegInsertionSide-v1 2\n" \
  | xargs -P 2 -n 2 bash -c 'run_one "$0" "$1"'

echo "[$(date -Is)] ALL DONE"
