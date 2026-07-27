#!/usr/bin/env bash
# Serve docs viewer on all interfaces (same Wi‑Fi / LAN).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d dist ]]; then
  echo "Building…"
  npm run build
fi

IP=$(hostname -I | awk '{print $1}')
HOST=$(hostname)

echo ""
echo "  NeuralProbes docs — LAN access"
echo "  ─────────────────────────────"
echo "  This machine:  http://127.0.0.1:4173/"
echo "  Same network:  http://${IP}:4173/"
echo "  Hostname:      http://${HOST}:4173/  (if DNS resolves on your LAN)"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

npm run preview:lan
