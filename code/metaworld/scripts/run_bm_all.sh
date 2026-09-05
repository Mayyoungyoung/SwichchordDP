#!/bin/bash
# Boundary Monitor + Best-of-K 完整实验矩阵:
#   3 个技能边界 × 2 个独立种子段, 每段 n=240
#   r2g  (reach→grasp→lift,  settle+lift×10 期望口径)
#   l2c  (lift→carry,  carry 语义)
#   c2p  (carry→place, place 语义)
set -u
cd "$(dirname "$0")/.."
PY=/home/jia/miniconda3/envs/swdp/bin/python
for pair in r2g l2c c2p; do
  for seed0 in 8000 9000; do
    echo "=== RUN pair=$pair seed0=$seed0 n=240 ==="
    CUDA_VISIBLE_DEVICES=0 $PY -u dc_eval_bm.py --pair "$pair" \
      --n_episodes 240 --k 3 --seed0 "$seed0" \
      --out "dc_bm_${pair}_s${seed0}"
  done
done
echo "ALL DONE"