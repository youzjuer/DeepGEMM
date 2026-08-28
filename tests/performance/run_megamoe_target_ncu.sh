#!/usr/bin/env bash
set -euo pipefail

side=${1:?usage: run_megamoe_target_ncu.sh official|candidate [num_tokens]}
num_tokens=${2:-32}
case "${side}" in
    official|candidate) ;;
    *) echo "invalid side: ${side}" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
official_repo=${DG_OFFICIAL_REPO:-/home/youchunbo/code/DeepGEMM_official_559d79f}
output_prefix=${DG_NCU_OUTPUT_PREFIX:-/tmp/dg_ncu_${side}_target_m${num_tokens}}
jit_cache=${DG_JIT_CACHE_DIR:-/tmp/dg_profile_m32_jit_1786510000/cache}
default_metrics=$(
    printf '%s' \
        'gpu__time_duration.sum,' \
        'smsp__inst_executed.sum,' \
        'smsp__inst_executed_pipe_tmem.sum,' \
        'smsp__inst_executed_pipe_tensor.sum,' \
        'smsp__inst_executed_pipe_tma.sum,' \
        'smsp__inst_executed_op_tma_ld.sum,' \
        'smsp__inst_executed_op_tma_st.sum,' \
        'smsp__inst_executed_op_shared_ld.sum,' \
        'smsp__inst_executed_op_shared_st.sum,' \
        'smsp__inst_executed_op_global_ld.sum,' \
        'smsp__inst_executed_op_global_st.sum,' \
        'smsp__inst_executed_op_generic_atom.sum'
)
metrics=${DG_NCU_METRICS:-${default_metrics}}
section=${DG_NCU_SECTION:-}

rm -f "${output_prefix}.ncu-rep" "${output_prefix}.json"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DG_JIT_CACHE_DIR="${jit_cache}"
export MASTER_ADDR=${DG_NCU_MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${DG_NCU_MASTER_PORT:-18361}

common_args=(
    tests/performance/bench_nvfp4_mode4_vs_official_ep8.py
    --num-processes 8
    --official-repo "${official_repo}"
    --num-tokens "${num_tokens}"
    --seed 20260812
    --warmup 0
    --repeat 0
    --stress-iterations 0
    --ncu-profile-only "${side}"
    --json-output "${output_prefix}.json"
)

rank_pids=()
cleanup() {
    for pid in "${rank_pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for rank in $(seq 1 7); do
    python3 "${common_args[@]}" --external-rank "${rank}" \
        > "${output_prefix}.rank${rank}.log" 2>&1 &
    rank_pids+=("$!")
done

profile_args=(
    --target-processes application-only
    --replay-mode "${DG_NCU_REPLAY_MODE:-kernel}"
    --kernel-name-base demangled
    --kernel-name 'regex:.*sm100_fp8_fp4_mega_moe_impl.*'
    --launch-count 1
)
if [[ -n "${section}" ]]; then
    profile_args+=(
        --section "${section}"
        --pm-sampling-interval "${DG_NCU_PM_SAMPLING_INTERVAL:-1000}"
        --pm-sampling-max-passes "${DG_NCU_PM_SAMPLING_MAX_PASSES:-1}"
    )
else
    profile_args+=(--metrics "${metrics}")
fi

set +e
ncu \
    "${profile_args[@]}" \
    --force-overwrite \
    --export "${output_prefix}" \
    python3 "${common_args[@]}" --external-rank 0
status=$?
for pid in "${rank_pids[@]}"; do
    wait "${pid}" || status=$?
done
set -e
rank_pids=()
exit "${status}"
