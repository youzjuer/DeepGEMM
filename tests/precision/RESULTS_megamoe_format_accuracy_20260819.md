# MegaMoE MXFP8×MXFP4 vs NVFP4×NVFP4 format accuracy

## Conclusion

There is no distribution-independent winner.

- For normal inputs, activation outliers, and the block-skew stress case,
  MXFP8×MXFP4 has lower output error.
- When weight outliers dominate, or both operands are heavy-tailed,
  NVFP4×NVFP4 has lower output error. Its group-16 E4M3 scales adapt more
  locally than MXFP4's group-32 power-of-two scales.
- The result is stable across all 20 paired seeds: every reported row has a
  20/20 win count for the same format.

The model-level choice must therefore be made from captured target-model
weights and activations, not from format bit width alone.

## Method

- DeepGEMM commit: `5460bfbccdbfb1744d91d613ca03b7516b8ae344`
- MegaMoE dimensions: `H=4096`, `I=1024`, `M={32,160}`
- Source tensors: the same BF16-rounded values for both formats
- MX path: E4M3 activation + group-32 UE8M0 scale; E2M1 weight + group-32
  UE8M0 scale
- NV path: E2M1 activation and weight + group-16 E4M3 scale; implicit global
  scale of one, matching the current MegaMoE helper contract
- Reference: unquantized BF16-rounded operands with FP32 accumulation
- Pipeline: GEMM1 -> BF16 boundary -> SwiGLU -> route weight -> requantize ->
  GEMM2
- Metric: relative L2 after the candidate and reference outputs are rounded to
  BF16
- Scenarios: normal, activation outliers, weight outliers, both outliers,
  Student-t heavy tail, and block skew
- Seeds: 0 through 19, paired between formats

Before the experiment, the standalone format implementation was checked
element-by-element against DeepGEMM's FP8, MXFP4, and NVFP4 quantize/dequantize
helpers. The maximum absolute difference was zero for all three.

## Representative pipeline result (`M=160`)

`NV/MX` is the paired median of `NV rel-L2 / MX rel-L2`. A value greater than
one means MX has lower error.

| Scenario | MX rel-L2 | NV rel-L2 | NV/MX | Lower error |
| --- | ---: | ---: | ---: | --- |
| normal | 0.205707 | 0.243683 | 1.186 | MX, NV error is 18.6% higher |
| activation outlier | 0.206849 | 0.233724 | 1.129 | MX, NV error is 12.9% higher |
| block skew | 0.204801 | 0.237863 | 1.156 | MX, NV error is 15.6% higher |
| weight outlier | 0.270618 | 0.218296 | 0.807 | NV, error is 19.3% lower |
| both outlier | 0.270179 | 0.194988 | 0.721 | NV, error is 27.9% lower |
| heavy tail | 0.266086 | 0.234641 | 0.881 | NV, error is 11.9% lower |

For the normal `M=160` case, the same direction is visible at each stage:

| Stage | MX rel-L2 | NV rel-L2 | NV/MX |
| --- | ---: | ---: | ---: |
| GEMM1 | 0.118393 | 0.143873 | 1.215 |
| GEMM2 | 0.118261 | 0.136271 | 1.152 |
| SwiGLU pipeline | 0.205707 | 0.243683 | 1.186 |

## Reproduction

```bash
python tests/precision/compare_mxfp8_mxfp4_vs_nvfp4.py \
  --device cuda:0 \
  --hidden 4096 \
  --intermediate 1024 \
  --m-values 32 160 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --scenarios normal activation_outlier weight_outlier both_outlier heavy_tail block_skew \
  --output /tmp/megamoe_precision.json \
  --csv /tmp/megamoe_precision.csv
```

The recorded run used PyTorch 2.13 on CPU and took 79.4 seconds. This is valid
for the format-level oracle because all products are formed from explicitly
dequantized FP32 tensors. It does not measure tensor-core accumulation order,
fast-math, fused epilogues, routing, or EP communication. Those require a
separate L20D kernel validation on an explicitly authorized Pod.
