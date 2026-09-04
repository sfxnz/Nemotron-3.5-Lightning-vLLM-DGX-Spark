# AGENTS.md — Nemotron 3.5 Lightning · 1× DGX Spark

Serve `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` on one Spark with stock `vllm/vllm-openai:v0.27.1`. Hybrid Mamba-2 + MoE, NVFP4, optional DSpark (3 draft tokens). Container `spark-nemotron-lightning` on port 8000.

Humans read [README.md](README.md). Decode ~93 tok/s is the L.A.I.L lab bench (run id `20260812T131332Z_d0f7ae`), not the 2× Spark `bench_decode.py` harness. This repo has neither that harness nor unit tests.

## Working rules

- One-node recipe. No worker SSH, no NCCL pin, no `recipe.yaml`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Keep `--moe-backend marlin`, `--quantization modelopt_mixed`, `--kv-cache-dtype fp8`.
- Keep `CUTE_DSL_ARCH=sm_121a` and `TORCH_CUDA_ARCH_LIST=12.1a`.
- Default `UTIL=0.8` and `MAX_MODEL_LEN=262144` left ~15 GiB available. Card-official 0.91 + 1M pushed free RAM toward ~1 GiB and used swap. Do not silently raise util to the card value on a shared box.
- `ENABLE_DSPARK=0` skips the draft when you need RAM.
- Thinking off for short generate: `chat_template_kwargs.enable_thinking=false`.

## Verify

```bash
./run.sh
curl -s http://127.0.0.1:8000/v1/models
```

Smoke is the README chat completion with `enable_thinking: false`. Do not invent a 2× Spark prose/structured table for this recipe.

## Never touch

- Live HF tokens
- Claiming SGLang ~112 tok/s as this recipe's number. This is the vLLM path.
