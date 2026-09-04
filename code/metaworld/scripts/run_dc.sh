#!/usr/bin/env bash
# Downstream-Compatible Skill Learning 主实验 (paper_plan_READY.md §5, grasp→lift)
set -e
cd "$(dirname "$0")/../../.."   # 仓库根目录
source /home/jia/miniconda3/bin/activate swdp
cd code/metaworld

# 1. 数据: 脚本 reach + DP grasp 240 条自然轨迹, 每条带冻结 lift ×10 outcome 标签
python dc_collect.py --n_episodes 240 --k_lift 10

# 2. 训练矩阵: uniform(等权微调对照) / outcome(上界, λ∈{1,2,4}) / quality(对照基线)
python dc_train.py --weight uniform --lam 0
python dc_train.py --weight outcome --lam 1.0
python dc_train.py --weight outcome --lam 2.0
python dc_train.py --weight outcome --lam 4.0
python dc_train.py --weight quality --lam 2.0

# 3. 评估: 每模型 120 rollout, 下游 lift 固定用冻结 base 模型;
#    评估种子 5000 起(与收集的 3000 起不相交, 留出集)
python dc_eval.py --n_episodes 120 --seed0 5000 \
  --models base,dc_uniform_l0.0,dc_outcome_l1.0,dc_outcome_l2.0,dc_outcome_l4.0,dc_quality_l2.0

echo "[run_dc] done"
