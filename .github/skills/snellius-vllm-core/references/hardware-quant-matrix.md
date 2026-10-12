# Snellius GPU Hardware / Quantization Compatibility

vLLM recipes at recipes.vllm.ai are written for whatever hardware the model
author tested on (often H200, MI300X, or Blackwell). Snellius has neither —
this table maps a recipe's requirements onto what's actually available, per
`snellius-core`'s partition specs.

| Partition | GPU | Architecture | VRAM/GPU | GPUs/node | Native FP8 | NVFP4 | INT4/AWQ/GPTQ |
|---|---|---|---|---|---|---|---|
| `gpu_a100` | NVIDIA A100 | Ampere | 40 GiB | 4 | No | No | Yes |
| `gpu_h100` | NVIDIA H100 | Hopper | 94 GiB | 4 | Yes | No | Yes |

Snellius has no Blackwell-generation GPUs, so **NVFP4 is never available**,
regardless of partition.

## Applying this during the interview

1. Take the recipe's list of quantization variants (e.g. `default` (bf16),
   `fp8`, `nvfp4`, `awq`, `gptq`).
2. Drop any variant the chosen partition can't run natively:
   - On `gpu_a100`: drop `fp8` and `nvfp4` recipe variants that assume
     hardware FP8 support — quantized `INT4`/`AWQ`/`GPTQ` checkpoints still
     work (they're software dequantization, not hardware FP8), as does the
     bf16/default checkpoint.
   - On `gpu_h100`: drop `nvfp4` only.
3. If the recipe's *recommended* variant was dropped, tell the user which
   variant is being substituted and why (e.g. "the recipe recommends the
   NVFP4 checkpoint for Blackwell; Snellius doesn't have Blackwell GPUs, so
   I'm using the FP8 checkpoint on `gpu_h100` instead").
4. VRAM sizing: total model VRAM (from the recipe's per-variant VRAM figure)
   must fit across `gpu_count × VRAM/GPU` for the chosen partition. If it
   doesn't fit even at 4 GPUs, say so — multi-node deployments are out of
   scope for these skills.
