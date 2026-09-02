#!/usr/bin/env bash
# 组合评测: baseline 对比 + 消融矩阵
set -e
cd "$(dirname "$0")/../../.."
source /home/jia/miniconda3/bin/activate swdp
cd code/metaworld
export CUDA_VISIBLE_DEVICES=0

SCENE=pick-place-v3
EP=12

# ---- A. baseline 对比(完整配置: mask + proj, lambda 已调优为 0.15) ----
for mode in chord naive eff_shift energy; do
  python eval_compose.py --scene $SCENE --mode $mode --n_episodes $EP --use_mask --use_proj --lam 0.15
done
# 端到端基线(需先训练)
python train_dp.py --scene ${SCENE}_full --n_iter 120000
python eval_e2e.py --scene $SCENE --n_episodes $EP

# ---- B. 消融: 时间掩码 ----
python eval_compose.py --scene $SCENE --mode chord --n_episodes $EP --use_proj --lam 0.15 --out chord_nomask

# ---- C. 消融: 可行性投影 ----
python eval_compose.py --scene $SCENE --mode chord --n_episodes $EP --use_mask --lam 0.15 --out chord_noproj

# ---- D. 消融: 噪声样本数 ----
for n in 2 4; do
  python eval_compose.py --scene $SCENE --mode chord --n_episodes $EP --use_mask --use_proj --lam 0.15 --n_noise $n --out chord_n${n}
done

# ---- E. 消融: delta ----
for d in 0.05 0.10 0.20 0.30; do
  python eval_compose.py --scene $SCENE --mode chord --n_episodes $EP --use_mask --use_proj --lam 0.15 --delta $d --out chord_d${d}
done

# ---- F. 消融: lambda(步长缩放) ----
for l in 0.3 1.0; do
  python eval_compose.py --scene $SCENE --mode chord --n_episodes $EP --use_mask --use_proj --lam $l --out chord_l${l}
done

# ---- 汇总 ----
python summarize.py --scene $SCENE
echo "[run_eval_all] done"
