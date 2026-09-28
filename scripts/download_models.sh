#!/usr/bin/env bash
# Downloads the MediaPipe Tasks model bundles into data/models/.
# These are Google's official hosted models (same URLs the mediapipe-samples repo uses).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/data/models"
mkdir -p "$DIR"

BASE="https://storage.googleapis.com/mediapipe-models"
fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then echo "exists: $out"; return; fi
  echo "downloading $(basename "$out")"
  curl -fL --retry 3 -o "$out" "$url"
}

fetch "$BASE/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" "$DIR/hand_landmarker.task"
fetch "$BASE/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" "$DIR/pose_landmarker_lite.task"
echo "done -> $DIR"
