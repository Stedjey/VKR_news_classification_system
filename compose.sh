#!/usr/bin/env bash
set -euo pipefail

docker compose down --remove-orphans
docker compose up --build -d
docker compose ps
