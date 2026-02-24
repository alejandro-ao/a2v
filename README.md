# audio-to-video

CLI tool to generate MP4 videos from audio with visual presets.

Implemented presets:
- `oscilloscope-3d`
- `kinetic-lyrics` (requires timed lyrics in `.lrc`)

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

Common options:

- `--width` (default `1920`)
- `--height` (default `1080`)
- `--fps` (default `30`)
- `--sample-rate` (default `22050`)
- `--font /path/to/font.ttf` (for lyric text rendering)

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
