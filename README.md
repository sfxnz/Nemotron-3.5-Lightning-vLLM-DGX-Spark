# Nemotron 3.5 Lightning · vLLM · DGX Spark (GB10)

**Easy one-command recipe** to run  
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)  
on a **single NVIDIA DGX Spark** with stock **vLLM 0.27.1**.

Hybrid Mamba-2 + MoE · **30B total / ~3B active** · **NVFP4** · optional **DSpark** speculative decoding.

---

## Measured on 1× DGX Spark (L.A.I.L lab)

| Metric | Result |
|--------|--------|
| Decode (c=1, median) | **~93 tok/s** |
| Prefill (c=1, median) | **~678 tok/s** |
| TTFT p50 | **~246 ms** |
| TPOT p50 | **~10.7 ms** |
| Mixed agent bench | **20/20 OK** |

Stack for those numbers: **vLLM 0.27.1**, util **0.8**, max-len **262k**, **DSpark** (3 draft tokens), marlin MoE, ModelOpt mixed, FP8 KV.  
Run id: `20260812T131332Z_d0f7ae`.

> Other public Spark numbers use different engines (e.g. SGLang ~112 tok/s single stream). This recipe is the **vLLM path** that works out of the box on ARM64 Spark.

---

## Requirements

- **DGX Spark (GB10)** with Docker + NVIDIA Container Toolkit
- ~**25–35 GiB** free disk for weights (+ draft)
- Prefer **≥15 GiB free system RAM** after load (this recipe uses `gpu-memory-utilization 0.8`)
- Hugging Face access (public model; token recommended for rate limits)

```bash
# Optional
hf auth login
# or: export HF_TOKEN=hf_...
```

---

## Quick start

```bash
git clone https://github.com/sfxnz/Nemotron-3.5-Lightning-vLLM-DGX-Spark.git
cd Nemotron-3.5-Lightning-vLLM-DGX-Spark

chmod +x run.sh stop.sh
./run.sh
```

First run:

1. Pulls `vllm/vllm-openai:v0.27.1` if needed  
2. Downloads main + DSpark draft weights into `~/.cache/huggingface` (resumable)  
3. Starts container **`spark-nemotron-lightning`** on **port 8000**

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq .
```

Stop:

```bash
./stop.sh
```

---

## What the script runs

Defaults (edit `run.sh` or env vars — see below):

| Setting | Default |
|---------|---------|
| Image | `vllm/vllm-openai:v0.27.1` |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| Draft (DSpark) | `…-NVFP4-DSpark` |
| `--gpu-memory-utilization` | **0.8** (safer than card’s 0.91 on a shared Spark) |
| `--max-model-len` | **262144** |
| `--quantization` | `modelopt_mixed` |
| `--kv-cache-dtype` | `fp8` |
| `--moe-backend` | `marlin` |
| `--max-num-seqs` | `4` |
| Reasoning / tools | `nemotron_v3` / `qwen3_coder` + auto tool choice |
| Speculative | DSpark, 3 draft tokens |
| Mamba | `flashinfer` + `align` |
| Arch env | `CUTE_DSL_ARCH=sm_121a`, `TORCH_CUDA_ARCH_LIST=12.1a` |

Card-official util is often **0.91** and context **up to 1M**. On a lab box that also runs agents/UI, **0.8 + 262k** left ~15 GiB available in our tests; **0.91** pushed free RAM toward ~1 GiB and used swap.

---

## Environment knobs

```bash
export HF_TOKEN=hf_...              # optional
export PORT=8000
export UTIL=0.8                     # try 0.7 if you need more free RAM
export MAX_MODEL_LEN=262144         # try 65536 for more headroom
export ENABLE_DSPARK=1              # set 0 to skip draft model
export CONTAINER_NAME=spark-nemotron-lightning
export VLLM_IMAGE=vllm/vllm-openai:v0.27.1

./run.sh
```

**Lab-friendly (more free RAM):**

```bash
UTIL=0.7 MAX_MODEL_LEN=65536 ENABLE_DSPARK=0 ./run.sh
```

**Closer to NVIDIA Spark card (aggressive):**

```bash
UTIL=0.91 MAX_MODEL_LEN=1048576 ./run.sh
```

---

## OpenAI-compatible API

| | |
|--|--|
| Base URL | `http://127.0.0.1:8000/v1` |
| Model id | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |

Thinking (when you want it):

```json
"chat_template_kwargs": { "enable_thinking": true }
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Temporary failure in name resolution` during download | Network/DNS glitch — re-run `./run.sh` (HF cache resumes) |
| OOM / only ~1 GiB free | Lower `UTIL` (0.7), lower `MAX_MODEL_LEN`, or `ENABLE_DSPARK=0` |
| Port in use | `PORT=8001 ./run.sh` or `./stop.sh` first |
| Container already exists | `./stop.sh` then `./run.sh` |
| Slow first boot | Waiting on HF download (~20 GiB main + draft) |

Logs:

```bash
docker logs -f spark-nemotron-lightning
```

---

## Manual docker (no script)

```bash
# weights (once)
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark

docker pull vllm/vllm-openai:v0.27.1

docker run -d --name spark-nemotron-lightning --gpus all --shm-size=4g \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  vllm/vllm-openai:v0.27.1 \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --trust-remote-code \
  --quantization modelopt_mixed \
  --kv-cache-dtype fp8 \
  --moe-backend marlin \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  --speculative_config.method dspark \
  --speculative_config.num_speculative_tokens 3 \
  --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark
```

---

## References

- Model card: [NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- DSpark draft: […-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)
- vLLM image: [`vllm/vllm-openai:v0.27.1`](https://hub.docker.com/r/vllm/vllm-openai)
- Lab automation that produced this recipe: [L.A.I.L](https://github.com/sfxnz/L.A.I.L)

---

## License

Recipe scripts: MIT.  
Model weights: see NVIDIA’s model license on Hugging Face ([OpenMDW-1.1](https://openmdw.ai/license/1-1/) as listed on the card).
