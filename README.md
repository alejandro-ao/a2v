# audio-to-video

CLI tool to generate MP4 videos from audio with visual presets.

Implemented presets:
- `oscilloscope-3d`
- `kinetic-lyrics` (animated captions from `.lrc` or auto-Whisper transcription)

## Quick Start

- `ffmpeg` installed and available in your `PATH`

macOS:

```bash
brew install ffmpeg
```

```bash
uv sync
```

```bash
uv run a2v presets
```

Fast preview on long audio (recommended while iterating styles):
```bash
scripts/quick-preview.sh \
  -i /Users/alejandro/Downloads/The_Hidden_Grammar_of_Woof_PDF.m4a \
  -s 00:02:10 \
  -d 10 \
  -o ./preview-10s.mp4
```

Full render with auto captions (local Whisper):
```bash
uv run a2v render ./input/song.mp3 -o ./output/song-captions.mp4 \
  --preset kinetic-lyrics \
  --whisper-device cpu \
  --whisper-compute-type int8 \
  --whisper-model base \
  --overwrite
```

Full render with manual `.lrc` captions:
```bash
uv run a2v render ./input/song.mp3 -o ./output/song-lyrics.mp4 \
  --preset kinetic-lyrics \
  --lyrics ./input/song.lrc \
  --overwrite
```

Note: first Whisper run downloads the selected model.

## AI Skill Install

Install this repo as a remote skill collection:

```bash
npx skills add alejandro-ao/a2v
```

Use the skill at:
- `skills/audio-to-video`

## Useful Options

- `--width` (default `1920`)
- `--height` (default `1080`)
- `--fps` (default `30`)
- `--sample-rate` (default `22050`)
- `--font /path/to/font.ttf` (for lyric text rendering)
- `--transcribe/--no-transcribe` (default: transcribe when `--lyrics` is not provided)
- `--whisper-model` (default `small`)
- `--whisper-language en|es|...` (optional hint)
- `--whisper-device auto|cpu|cuda`
- `--whisper-compute-type default|int8|float16|...`

`scripts/quick-preview.sh` defaults optimized for fast local iteration:
- `--model tiny`
- `--device cpu`
- `--compute-type int8`
- output `1280x720` at `30fps`

## LRC format

Use timestamped lines:

```text
[00:01.20]First lyric line
[00:04.90]Second lyric line
[00:09.40]Third lyric line
```

Multiple timestamps on one line are supported:

```text
[00:10.00][00:45.00]Same line appears twice
```
