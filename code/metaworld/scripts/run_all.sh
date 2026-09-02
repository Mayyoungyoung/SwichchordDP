#!/usr/bin/env bash
# SWDP Meta-World 概念验证一键脚本
set -e
cd "$(dirname "$0")/../../.."   # 仓库根目录
source /home/jia/miniconda3/bin/activate swdp
cd code/metaworld

SCENE=pick-place-v3

# 1. 数据采集(技能演示 + 完整任务演示)
python make_dataset.py --scene $SCENE --n_demos 30
python collect_full.py --scene $SCENE --n_demos 40

# 2. 训练技能条件 DP 与端到端基线 DP
python train_dp.py --scene $SCENE --n_iter 60000
python train_dp.py --scene ${SCENE}_full --n_iter 60000 --data_dir results/metaworld/data

# 3. 组合评测: baseline 对比
for mode in chord naive eff_shift; do
  python eval_compose.py --scene $SCENE --mode $mode --n_episodes 10 --use_mask --use_proj
done
# 无掩码/无投影消融
python eval_compose.py --scene $SCENE --mode chord --n_episodes 10

# 4. 端到端基线
python eval_e2e.py --scene $SCENE --n_episodes 10

echo "[run_all] done"
