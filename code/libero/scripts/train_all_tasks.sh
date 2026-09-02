#!/usr/bin/env bash
# 每任务独立训练 LIBERO 技能 DP
set -e
cd "$(dirname "$0")/../../.."
source /home/jia/miniconda3/bin/activate turbovla-libero
cd code/libero
for f in ../../results/libero/data_replay/*.h5; do
  task=$(basename "$f" .h5)
  echo "=== train $task ==="
  CUDA_VISIBLE_DEVICES=1 python train_libero_replay.py --task "$task" --n_iter 60000 --tau_power 2.0
done
echo "[train_all_tasks] done"
