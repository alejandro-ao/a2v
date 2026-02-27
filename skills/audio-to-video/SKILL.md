---
name: audio-to-video
description: Generate MP4 videos from audio using the local `a2v` CLI. Use when asked to turn audio into video, create animated captions from `.lrc` or Whisper, produce oscilloscope visualizers, or iterate quickly with short preview renders before final export.
---

# Audio To Video

Use this workflow:

1. Ensure dependencies are available.
```bash
brew install ffmpeg
uv sync
```

2. For long recordings, iterate with a short preview first.
```bash
scripts/quick-preview.sh -i <audio-file> -s 00:02:10 -d 10 -o ./preview.mp4
```

3. Pick rendering mode:
- `kinetic-lyrics`: spoken-word captions with active-word color highlight.
- `oscilloscope-3d`: waveform visualizer, no captions.

4. Render final output.
```bash
uv run a2v render <audio-file> -o <output.mp4> -p kinetic-lyrics \
  --whisper-device cpu --whisper-compute-type int8 --whisper-model base --overwrite
```

5. Use manual captions instead of Whisper when needed.
```bash
uv run a2v render <audio-file> -o <output.mp4> -p kinetic-lyrics \
  --lyrics <captions.lrc> --overwrite
```

Notes:
- Whisper runs locally in this project.
- First Whisper run downloads model weights.
- On Apple Silicon, prefer `--whisper-device cpu --whisper-compute-type int8`.
