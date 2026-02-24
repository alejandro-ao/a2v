# audio-to-video

CLI tool to generate MP4 videos from audio with visual presets.

Implemented presets:
- `oscilloscope-3d`
- `kinetic-lyrics` (animated captions from `.lrc` or auto-Whisper transcription)

## Requirements

- `ffmpeg` installed and available in your `PATH`

macOS:

```bash
brew install ffmpeg
```

## Install deps

```bash
uv sync
```

Note: first Whisper run downloads the selected model.

## Commands

List presets:

```bash
uv run a2v presets
```

Render `oscilloscope-3d`:

```bash
uv run a2v render ./input/song.mp3 -o ./output/song-osc.mp4 --preset oscilloscope-3d --overwrite
```

Render `kinetic-lyrics`:

```bash
uv run a2v render ./input/song.mp3 -o ./output/song-lyrics.mp4 --preset kinetic-lyrics --lyrics ./input/song.lrc --overwrite
```

Render `kinetic-lyrics` with auto-generated captions (Whisper):

```bash
uv run a2v render ./input/song.mp3 -o ./output/song-captions.mp4 --preset kinetic-lyrics --whisper-model base --overwrite
```

Common options:

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
