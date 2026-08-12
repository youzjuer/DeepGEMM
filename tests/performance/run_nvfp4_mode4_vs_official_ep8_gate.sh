#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_dir}"

official_base="559d79fb6994a58b8a15b4b93bf13ccc16edf247"
official_repo="${OFFICIAL_REPO:-/home/youchunbo/code/DeepGEMM_official_559d79f}"
test "$(git -C "${official_repo}" rev-parse HEAD)" = "${official_base}"

result_dir="${RESULT_DIR:-/tmp/deepgemm_nvfp4_mode4_ep8_gate_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${result_dir}"

export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export DG_JIT_CACHE_DIR="${DG_JIT_CACHE_DIR:-/tmp/deepgemm_nvfp4_mode4_ep8_gate_jit}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
candidate_contract="${CANDIDATE_CONTRACT:-mxfp4}"

for mask in 0.5 1.0; do
  tag="${mask/./}"
  python3 tests/performance/bench_nvfp4_mode4_vs_official_ep8.py \
    --num-processes 8 \
    --official-repo "${official_repo}" \
    --candidate-contract "${candidate_contract}" \
    --num-tokens 32 \
    --seed 20260814 \
    --masked-route-fraction "${mask}" \
    --warmup 0 \
    --repeat 0 \
    --stress-iterations "${STRESS_ITERATIONS:-1000}" \
    --cuda-graph-replays "${CUDA_GRAPH_REPLAYS:-1000}" \
    --json-output "${result_dir}/stress_mask${tag}.json"
done

shapes=(${SHAPES:-32 64 96 128 160})
seeds=(${SEEDS:-20260814 20260815 20260816})
for shape in "${shapes[@]}"; do
  for seed in "${seeds[@]}"; do
    python3 tests/performance/bench_nvfp4_mode4_vs_official_ep8.py \
      --num-processes 8 \
      --official-repo "${official_repo}" \
      --candidate-contract "${candidate_contract}" \
      --num-tokens "${shape}" \
      --seed "${seed}" \
      --warmup "${WARMUP:-20}" \
      --repeat "${REPEAT:-100}" \
      --json-output "${result_dir}/matrix_m${shape}_seed${seed}.json"
  done
done

python3 tests/performance/summarize_nvfp4_mode4_vs_official_ep8.py \
  "${result_dir}"/matrix_m*_seed*.json \
  --stress-inputs "${result_dir}"/stress_mask*.json \
  --expected-seeds "${#seeds[@]}" \
  --output "${result_dir}/summary.json"

echo "RESULT_DIR=${result_dir}"
