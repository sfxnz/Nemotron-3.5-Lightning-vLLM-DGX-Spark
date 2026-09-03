# Nemotron 3.5 Lightning · vLLM · 1× DGX Spark

Serve [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) on one NVIDIA DGX Spark (GB10) at tensor-parallel 1.

30B total / 3B active. Hybrid Mamba-2 + MoE. NVFP4 (ModelOpt mixed, fp8 KV). Native context is 1,048,576. This recipe serves `--max-model-len` 262144 with the DSpark drafter (3 speculative tokens).

Stock `vllm/vllm-openai:v0.27.1` is the arm64 image that runs on sm_121. `docker pull` it. There is no local `docker/` chain.

Pinned snapshot: `cc84af2fe71647d87f4486c064f320e1e7535243`. DSpark draft [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark) is pinned to `d10c6ff40d6e69d1f92e407e027de3eafdb77645` (`DRAFT_SHA`); `run.sh` downloads it once next to the main weights and serves it from that snapshot.

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
| Draft | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` @ `d10c6ff40d6e69d1f92e407e027de3eafdb77645` |
| Reasoning / tools | `nemotron_v3` / `qwen3_coder` + auto tool choice (in `EXTRA_ARGS`) |
| Mamba | `flashinfer` + `align` (in `EXTRA_ARGS`) |
| API | `http://<head>:8000/v1` |
| Container | `nemotron-3-5-lightning-vllm-dgx-spark` |
| Master port | 29500 |
<!-- END generated defaults -->

`run.sh` refuses `--max-model-len` above 262144 unless `FORCE_UNSAFE_CTX=1`. Native / card-official window is 1,048,576. Lab default stays 262144 at util 0.80. `FORCE_UNSAFE_CTX=1 MAX_MODEL_LEN=1048576 ./run.sh` is the override.

## Measured on 1× DGX Spark (L.A.I.L lab)

Decode only. Streamed greedy, thinking off, 200 completion tokens max, 3-run median. util 0.80, fp8 KV, context 262144, DSpark-3, CUDA graphs. Prose is the low-acceptance regime. Structured (count 1→200) is the high-acceptance regime. `python3 kit/bench_decode.py --recipe . --phase both --out evidence/<unit>-<UTC>` repeats both phases at c=1,2 and writes `bench.txt` + `bench.json`.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 88.5 | 88.4 | 0.087 s |
| prose | 2 | 74.5 | 144.2 | 0.090 s |
| structured | 1 | 163.0 | 162.9 | 0.066 s |
| structured | 2 | 133.5 | 262.5 | 0.113 s |
<!-- END generated measured -->

Rows from `evidence/rehearsal-nemotron-20260902T234001Z/bench.json` (gate unit `rehearsal-nemotron`, ready after 287 s). Prose stopped at 106 completion tokens (EOS before the 200 cap); structured used all 200. Draft acceptance in the same file: prose 0.38, structured 0.99 (`draft_acceptance_rate`).

Needle: hit at 21855 prompt tokens, prefill 5742.8 tok/s, TTFT 3.806 s (`evidence/rehearsal-nemotron-20260902T234001Z/needle-8192.txt`; the probe targets 8192 tokens, the Nemotron tokenizer counts 21855). Do not copy numbers from an older README; they have no file under `evidence/`.

## Agent-readiness probes

From `evidence/rehearsal-nemotron-20260902T234001Z/probes.json` (`"failed": 0`). Each row names its receipt.

| Probe | Result |
|---|---|
| Smoke (`Say hello`, greedy) | PASS (`smoke.txt`) |
| Thinking off, no `<think>` leak in `content` | PASS (`thinking_off.txt`) |
| Tool call parsed (`get_weather`) | PASS (`tool_call.txt`) |
| Tool follow-up (`role: tool`) answers without a think leak | PASS (`hermes_two_turn.txt`) |
| Greedy count 1→200 consecutive | PASS (`count.txt`) |
| Unique-salt needle, 21855 prompt tokens | PASS (`needle-8192.txt`) |

## Evidence

Every number above has a file under [`evidence/`](evidence/). `recipe.yaml` names the file per measured row. `python3 kit/render.py --check` lists the rows that still have none. See [`evidence/README.md`](evidence/README.md).

## Gotchas

- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`.
- Stock `vllm/vllm-openai:v0.27.1` is the image. Do not swap in an untested tag.
- `run.sh` refuses `--max-model-len` above 262144 unless `FORCE_UNSAFE_CTX=1`.
- DSpark draft is a second download, pinned by `DRAFT_SHA` (not `SNAPSHOT_SHA`). Its cache dir may be root-owned from an in-container download; `run.sh` never re-downloads a snapshot that exists.
- `SPEC_CONFIG` stays empty on purpose. JSON with double quotes in `serve.env` loses quotes in bash. Spec flags live in `EXTRA_ARGS` as dotted vLLM words.

## License

Recipe scripts are MIT. Model weights follow the base model license on Hugging Face (OpenMDW-1.1).
