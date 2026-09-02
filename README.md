# Nemotron 3.5 Lightning · vLLM · 1× DGX Spark

Serve [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) on one NVIDIA DGX Spark (GB10) at tensor-parallel 1.

30B total / 3B active. Hybrid Mamba-2 + MoE. NVFP4 (ModelOpt mixed, fp8 KV). Native context is 1,048,576. This recipe serves `--max-model-len` 262144 with the DSpark drafter (3 speculative tokens).

Stock `vllm/vllm-openai:v0.27.1` is the arm64 image that runs on sm_121. `docker pull` it. There is no local `docker/` chain.

Pinned snapshot: `cc84af2fe71647d87f4486c064f320e1e7535243`. DSpark draft [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark) is not pinned; `run.sh` downloads it next to the main weights.

## Hardware

- One DGX Spark (GB10) with Docker + NVIDIA Container Toolkit
- About 25–35 GiB free disk for the weights plus the DSpark draft
- Exclusive GPU. Do not start this recipe while another `--gpus all` serve is up.

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Quick start

```bash
docker pull vllm/vllm-openai:v0.27.1
chmod +x run.sh stop.sh
VALIDATE_ONLY=1 ./run.sh   # checks the defaults, no Docker
./run.sh
```

Single node (`NNODES=1`). The head does not copy itself to `spark2`.

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Stop:

```bash
./stop.sh
```

## Defaults

`recipe.yaml` is the source of truth. Edit it, then run `python3 kit/render.py`. CI fails when this table or the `run.sh` block drifts from it.

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.27.1` (stock, `docker pull`) |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| Checkpoint | `cc84af2fe71647d87f4486c064f320e1e7535243` |
| `--tensor-parallel-size` / `--nnodes` | 1 / 1 |
| `--max-model-len` | 262144 |
| `--max-num-seqs` | 4 |
| `--gpu-memory-utilization` | 0.80 |
| `--kv-cache-dtype` | `fp8` |
| `--kv-cache-memory` | not pinned yet (`KV_CACHE_MEMORY` is empty; vLLM decides) |
| `--block-size` | vLLM default (`BLOCK_SIZE` is empty) |
| CUDA graphs | on (`ENFORCE_EAGER=1` reverts to `--enforce-eager`) |
| Quantization / MoE | `modelopt_mixed` / `marlin` (in `EXTRA_ARGS`) |
| Speculative | DSpark, 3 draft tokens (in `EXTRA_ARGS`) |
| Draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` (not pinned) |
| Reasoning / tools | `nemotron_v3` / `qwen3_coder` + auto tool choice (in `EXTRA_ARGS`) |
| Mamba | `flashinfer` + `align` (in `EXTRA_ARGS`) |
| API | `http://<head>:8000/v1` |
| Container | `nemotron-3-5-lightning-vllm-dgx-spark` |
| Master port | 29500 |
<!-- END generated defaults -->

`run.sh` refuses `--max-model-len` above 262144 unless `FORCE_UNSAFE_CTX=1`. Native / card-official window is 1,048,576. Lab default stays 262144 at util 0.80. `FORCE_UNSAFE_CTX=1 MAX_MODEL_LEN=1048576 ./run.sh` is the override.

## Measured on 1× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens, 3-run median. util 0.80, fp8 KV, context 262144, DSpark-3, CUDA graphs. Prose is the low-acceptance regime. Structured (count 1→200) is the high-acceptance regime. `python3 kit/bench_decode.py --recipe . --phase both --out evidence/<unit>-<UTC>` repeats both phases at c=1,2 and writes `bench.txt` + `bench.json`.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | TODO | TODO | TODO s |
| prose | 2 | TODO | TODO | TODO s |
| structured | 1 | TODO | TODO | TODO s |
| structured | 2 | TODO | TODO | TODO s |
<!-- END generated measured -->

Prefill and needle results wait on the gate. Do not copy numbers from an older README; they have no file under `evidence/`.

## Agent-readiness probes

One result per probe after the gate. Each has a receipt under `evidence/`.

| Probe | Result |
|---|---|
| Thinking off, no `<think>` leak in `content` | TODO |
| Tool call parsed (`get_weather`) | TODO |
| Tool follow-up (`role: tool`) answers without a think leak | TODO |
| Greedy count 1→200 consecutive | TODO |
| Unique-salt needle at 8192 prompt tokens | TODO |

## Evidence

Every number above has a file under [`evidence/`](evidence/). `recipe.yaml` names the file per measured row. `python3 kit/render.py --check` lists the rows that still have none. See [`evidence/README.md`](evidence/README.md).

## Gotchas

- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.
- Stock `vllm/vllm-openai:v0.27.1` is the image. Do not swap in an untested tag.
- `run.sh` refuses `--max-model-len` above 262144 unless `FORCE_UNSAFE_CTX=1`.
- DSpark draft is a second download and is not pinned by `SNAPSHOT_SHA`.
- `SPEC_CONFIG` stays empty on purpose. JSON with double quotes in `serve.env` loses quotes in bash. Spec flags live in `EXTRA_ARGS` as dotted vLLM words.

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (OpenMDW-1.1).
