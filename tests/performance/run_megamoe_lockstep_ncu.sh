#!/usr/bin/env bash
set -euo pipefail

side=${1:?usage: run_megamoe_lockstep_ncu.sh official|candidate [num_tokens] [struct|pm]}
num_tokens=${2:-32}
profile_kind=${3:-struct}
case "${side}" in
    official|candidate) ;;
    *) echo "invalid side: ${side}" >&2; exit 2 ;;
esac
case "${profile_kind}" in
    struct|pm) ;;
    *) echo "invalid profile kind: ${profile_kind}" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
official_repo=${DG_OFFICIAL_REPO:-/home/youchunbo/code/DeepGEMM_official_559d79f}
output_prefix=${DG_NCU_OUTPUT_PREFIX:-/tmp/dg_ncu_${side}_${profile_kind}_m${num_tokens}}
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
ncu_profile_args=()
replay_mode=${DG_NCU_REPLAY_MODE:-kernel}
case "${profile_kind}" in
    struct)
        ncu_profile_args+=(--disable-extra-suffixes --metrics "${metrics}")
        ;;
    pm)
        replay_mode=${DG_NCU_REPLAY_MODE:-application}
        ncu_profile_args+=(
            --section PmSampling
            --pm-sampling-interval 1000
            --pm-sampling-max-passes 1
        )
        ;;
esac

rm -f "${output_prefix}.ncu-rep" "${output_prefix}.json"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
DG_JIT_CACHE_DIR="${jit_cache}" ncu \
    --target-processes all \
    --profile-from-start "${DG_NCU_PROFILE_FROM_START:-on}" \
    --communicator shmem \
    --communicator-shmem-num-peers 8 \
    --lockstep-kernel-launch \
    --replay-mode "${replay_mode}" \
    --kernel-name-base function \
    --kernel-name sm100_fp8_fp4_mega_moe_impl \
    --launch-skip 0 \
    --launch-count 1 \
    --cache-control none \
    --clock-control none \
    "${ncu_profile_args[@]}" \
    --force-overwrite \
    --export "${output_prefix}" \
    python3 tests/performance/bench_nvfp4_mode4_vs_official_ep8.py \
        --num-processes 8 \
        --official-repo "${official_repo}" \
        --num-tokens "${num_tokens}" \
        --seed 20260812 \
        --warmup 0 \
        --repeat 0 \
        --stress-iterations 0 \
        --ncu-profile-only "${side}" \
        --ncu-profile-from-start \
        --json-output "${output_prefix}.json"
