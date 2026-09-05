#!/bin/bash
# r2g 巩固 + 全链验证矩阵:
#   1) theta 灵敏度: 0.008 / 0.012 (s8000)
#   2) K 灵敏度:     1 / 5     (s8000)
#   3) 第三种子段:   theta=0.010 s10000
#   4) 全链 full:    theta=0.010 s8000 / s9000
set -u
cd "$(dirname "$0")/.."
PY=/home/jia/miniconda3/envs/swdp/bin/python

run() {  # $1=pair $2=seed0 $3=tag $4...=额外参数
  local pair=$1 seed0=$2 tag=$3; shift 3
  echo "=== RUN $tag ==="
  CUDA_VISIBLE_DEVICES=0 $PY -u dc_eval_bm.py --pair "$pair" \
    --n_episodes 240 --seed0 "$seed0" --out "dc_bm_${tag}" "$@"
}

run r2g 8000 r2g_s8000_th008  --thresh 0.008
run r2g 8000 r2g_s8000_th012  --thresh 0.012
run r2g 8000 r2g_s8000_k1    --k 1
run r2g 8000 r2g_s8000_k5    --k 5
run r2g 10000 r2g_s10000     --thresh 0.010
run full 8000 full_s8000     --thresh 0.010
run full 9000 full_s9000     --thresh 0.010
echo "ALL DONE"