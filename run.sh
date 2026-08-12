#!/usr/bin/env bash
# Nemotron 3.5 Lightning NVFP4 on DGX Spark — one-shot vLLM serve
set -euo pipefail

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
DRAFT="${DRAFT:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.27.1}"
CONTAINER_NAME="${CONTAINER_NAME:-spark-nemotron-lightning}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Prefer official hf CLI; fall back to huggingface-cli
hf_bin() {
  if command -v hf >/dev/null 2>&1; then
    echo hf
  elif command -v huggingface-cli >/dev/null 2>&1; then
    echo huggingface-cli
  else
    return 1
  fi
}

log() { printf '==> %s\n' "$*"; }

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  log "Removing existing container $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

log "Ensuring image $VLLM_IMAGE"
if ! docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
  docker pull "$VLLM_IMAGE"
fi

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  if HF=$(hf_bin); then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    log "Downloading $MODEL (resumes cache under $HF_CACHE)"
    if [[ "$HF" == "hf" ]]; then
      "$HF" download "$MODEL"
      if [[ "$ENABLE_DSPARK" == "1" ]]; then
        log "Downloading DSpark draft $DRAFT"
        "$HF" download "$DRAFT"
      fi
    else
      "$HF" download "$MODEL"
      if [[ "$ENABLE_DSPARK" == "1" ]]; then
        "$HF" download "$DRAFT"
      fi
    fi
  else
    log "No hf/huggingface-cli on PATH — skipping host download; vLLM will pull weights on first load"
  fi
fi

EXTRA_SPEC=()
if [[ "$ENABLE_DSPARK" == "1" ]]; then
  EXTRA_SPEC=(
    --speculative_config.method dspark
    --speculative_config.num_speculative_tokens 3
    --speculative_config.model "$DRAFT"
  )
fi

ENV_ARGS=(
  -e HF_HOME=/cache/huggingface
  -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a
)
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(-e "HF_TOKEN=$HF_TOKEN" -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN")
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
  TOK=$(tr -d '[:space:]' <"$HOME/.cache/huggingface/token" || true)
  if [[ -n "$TOK" ]]; then
    ENV_ARGS+=(-e "HF_TOKEN=$TOK" -e "HUGGING_FACE_HUB_TOKEN=$TOK")
  fi
fi

log "Starting $CONTAINER_NAME (util=$UTIL max-model-len=$MAX_MODEL_LEN dspark=$ENABLE_DSPARK)"
# Stock vllm-openai ENTRYPOINT is already `vllm serve`
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart no \
  --gpus all \
  --shm-size=4g \
  -p "127.0.0.1:${PORT}:8000" \
  -v "${HF_CACHE}:/cache/huggingface" \
  "${ENV_ARGS[@]}" \
  "$VLLM_IMAGE" \
  "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --quantization modelopt_mixed \
  --kv-cache-dtype fp8 \
  --moe-backend marlin \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --enable-prefix-caching \
  --mamba-backend flashinfer \
  --mamba-cache-mode align \
  "${EXTRA_SPEC[@]}"

log "Container started. Waiting for /v1/models …"
ok=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ok=1
    break
  fi
  # fail fast if container died
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container exited early. Logs:" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -80 >&2
    exit 1
  fi
  sleep 5
  if (( i % 6 == 0 )); then
    log "still loading… (${i}×5s) — docker logs -f $CONTAINER_NAME"
  fi
done

if [[ "$ok" != "1" ]]; then
  echo "Timed out waiting for API. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" 2>&1 | tail -100 >&2
  exit 1
fi

log "Ready → http://127.0.0.1:${PORT}/v1"
curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400 || true
echo
log "Stop with: ./stop.sh"
