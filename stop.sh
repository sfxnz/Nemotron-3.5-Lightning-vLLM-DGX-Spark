#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="${CONTAINER_NAME:-spark-nemotron-lightning}"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Stopping $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
  echo "Stopped."
else
  echo "No container named $CONTAINER_NAME"
fi
