#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Fast preview renderer for visual iteration.

Usage:
  scripts/quick-preview.sh -i INPUT_AUDIO [options]

Required:
  -i, --input PATH             Source audio file (full-length audio is fine)

Options:
  -o, --output PATH            Output mp4 path (default: ./preview.mp4)
  -s, --start TIME             Start offset for preview clip (default: 00:00:30)
  -d, --duration SECONDS       Clip duration in seconds (default: 10)
  -p, --preset NAME            Visual preset (default: kinetic-lyrics)
      --lyrics PATH            Optional LRC file for clip
      --model NAME             Whisper model for auto captions (default: tiny)
      --device DEVICE          Whisper device (default: cpu)
      --compute-type TYPE      Whisper compute type (default: int8)
      --width N                Output width (default: 1280)
      --height N               Output height (default: 720)
      --fps N                  Output fps (default: 30)
  -h, --help                   Show this help

Examples:
  scripts/quick-preview.sh \
    -i /Users/alejandro/Downloads/The_Hidden_Grammar_of_Woof_PDF.m4a \
    -s 00:02:10 -d 12 -o ./preview-12s.mp4

  scripts/quick-preview.sh \
    -i /Users/alejandro/Downloads/The_Hidden_Grammar_of_Woof_PDF.m4a \
    -p oscilloscope-3d -d 8 -o ./preview-osc.mp4
EOF
}

INPUT=""
OUTPUT="./preview.mp4"
START="00:00:30"
DURATION="10"
PRESET="kinetic-lyrics"
LYRICS=""
MODEL="tiny"
DEVICE="cpu"
COMPUTE_TYPE="int8"
WIDTH="1280"
HEIGHT="720"
FPS="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -s|--start)
      START="$2"
      shift 2
      ;;
    -d|--duration)
      DURATION="$2"
      shift 2
      ;;
    -p|--preset)
      PRESET="$2"
      shift 2
      ;;
    --lyrics)
      LYRICS="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --compute-type)
      COMPUTE_TYPE="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "Missing required --input." >&2
  usage
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input audio not found: $INPUT" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not found in PATH." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
CLIP_PATH="$TMP_DIR/preview-clip.m4a"

echo "[preview] Creating ${DURATION}s clip at ${START} from: $INPUT"
ffmpeg -hide_banner -loglevel error -y \
  -ss "$START" -t "$DURATION" \
  -i "$INPUT" -vn -c:a aac "$CLIP_PATH"

mkdir -p "$(dirname "$OUTPUT")"

CMD=(
  uv run a2v render "$CLIP_PATH"
  -o "$OUTPUT"
  --preset "$PRESET"
  --width "$WIDTH"
  --height "$HEIGHT"
  --fps "$FPS"
  --overwrite
)

if [[ -n "$LYRICS" ]]; then
  CMD+=(--lyrics "$LYRICS")
else
  CMD+=(
    --transcribe
    --whisper-model "$MODEL"
    --whisper-device "$DEVICE"
    --whisper-compute-type "$COMPUTE_TYPE"
  )
fi

echo "[preview] Rendering: $OUTPUT"
"${CMD[@]}"
echo "[preview] Done."

