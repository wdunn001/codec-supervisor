#!/usr/bin/env bash
# deploy-codec-metamcp-lab.sh — swap the production codec-metamcp container
# on the lab box (192.168.1.88) to a new image tag in place. Run via:
#
#   ssh vinez@192.168.1.88 'bash -s' < deploy-codec-metamcp-lab.sh v0.2.5
#
# Reads the existing container's env so the swap is exact: same DB, same
# network, same secrets, same APP_URL, etc. Doesn't write anything new.
set -euo pipefail

TAG="${1:-latest}"
NAME="metamcp"
NETWORK="open_web_ui_default"
PORT="12008"

if ! docker image inspect "wdunn001/codec-metamcp:${TAG}" >/dev/null 2>&1; then
    echo "Image wdunn001/codec-metamcp:${TAG} not found locally on the lab box."
    echo "Build it first via: docker build ... -t wdunn001/codec-metamcp:${TAG}"
    exit 1
fi

# Pull the existing container's env so the swap is byte-equivalent.
mapfile -t ENVS < <(docker inspect "${NAME}" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | grep -vE '^(PATH=|NODE_VERSION=|YARN_VERSION=)' \
    | grep -v '^$')

echo "Stopping ${NAME}..."
docker stop "${NAME}" >/dev/null
docker rm "${NAME}" >/dev/null

echo "Starting ${NAME} from wdunn001/codec-metamcp:${TAG}..."
ENV_ARGS=()
for e in "${ENVS[@]}"; do
    ENV_ARGS+=("-e" "${e}")
done

docker run -d \
    --name "${NAME}" \
    --network "${NETWORK}" \
    -p "${PORT}:${PORT}" \
    --restart unless-stopped \
    "${ENV_ARGS[@]}" \
    "wdunn001/codec-metamcp:${TAG}"

echo "Waiting for health..."
for i in $(seq 1 30); do
    if curl -fs -o /dev/null "http://localhost:${PORT}/api/health" 2>/dev/null \
        || curl -fs -o /dev/null "http://localhost:${PORT}/" 2>/dev/null; then
        echo "Up at http://localhost:${PORT}/ on iteration ${i}"
        break
    fi
    sleep 2
done

echo "---"
docker ps --filter "name=${NAME}" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
echo "---"
docker exec "${NAME}" cat /opt/codec/CODEC_METAMCP_SHA  || true
docker exec "${NAME}" cat /opt/codec/CODEC_METAMCP_HEAD || true
echo "---uvx---"
docker exec "${NAME}" /usr/local/bin/uvx --version || echo "uvx NOT installed"
