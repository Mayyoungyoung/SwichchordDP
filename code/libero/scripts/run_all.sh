#!/usr/bin/env bash
# SWDP LIBERO 增信实验一键脚本
set -e
cd "$(dirname "$0")/../../.."
source /home/jia/miniconda3/bin/activate turbovla-libero
cd code/libero

# 1. 分段提取
python extract_segments.py --task all
# 2. 训练技能条件 DP(低维)
CUDA_VISIBLE_DEVICES=0 python train_libero.py --task all --n_iter 40000
# 3. 离线拼接评测: chord vs naive vs eff_shift
for mode in chord naive eff_shift; do
  python eval_libero_offline.py --task all --mode $mode
done
python eval_libero_offline.py --task all --mode chord --use_proj

echo "[libero run_all] done"
