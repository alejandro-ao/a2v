from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import typer

from .audio import decode_audio_mono, require_binary
from .lyrics import parse_lrc
from .renderers import PRESET_KINETIC_LYRICS, PRESETS, RenderConfig, create_renderer

app = typer.Typer(
    name="a2v",
    help="Render audio-reactive MP4 videos from an audio file.",
    no_args_is_help=True,
)


@app.command("presets")
def cmd_presets() -> None:
    """List available visual presets."""
    print("Available presets:")
    for preset in PRESETS:
        print(f"- {preset}")

@app.command("render")
def cmd_render(
    input_audio: Path = typer.Argument(..., help="Input audio path (mp3/wav/m4a/...)."),
    output: Path = typer.Option(..., "-o", "--output", help="Output MP4 path."),
    preset: str = typer.Option(PRESETS[0], "-p", "--preset", help="Visual preset to render."),
    lyrics: Path | None = typer.Option(
        None,
        "--lyrics",
        help="Path to .lrc lyrics file (required for kinetic-lyrics).",
    ),
    font: Path | None = typer.Option(
        None,
        "--font",
        help="Optional path to a TTF/OTF/TTC font for lyric rendering.",
    ),
    fps: int = typer.Option(30, "--fps", help="Frames per second."),
    width: int = typer.Option(1920, "--width", help="Video width."),
    height: int = typer.Option(1080, "--height", help="Video height."),
    sample_rate: int = typer.Option(
        22_050,
        "--sample-rate",
        help="Audio sample rate used for analysis.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Render an MP4 video from an audio file."""
    if preset not in PRESETS:
        choices = ", ".join(PRESETS)
        raise typer.BadParameter(f"Invalid preset '{preset}'. Choose one of: {choices}.")

    ffmpeg = require_binary("ffmpeg")

    if not input_audio.exists():
        raise typer.BadParameter(f"Input audio does not exist: {input_audio}")
    if output.exists() and not overwrite:
        raise typer.BadParameter(
            f"Output already exists: {output}. Pass --overwrite to replace it."
        )
    if width < 64 or height < 64:
        raise typer.BadParameter("--width and --height must both be >= 64.")
    if fps < 1:
        raise typer.BadParameter("--fps must be >= 1.")

    print(f"Decoding audio: {input_audio}", file=sys.stderr)
    audio = decode_audio_mono(input_audio, sample_rate=sample_rate)

    parsed_lyrics = None
    if preset == PRESET_KINETIC_LYRICS:
        if lyrics is None:
            raise typer.BadParameter("--lyrics is required for preset 'kinetic-lyrics'.")
        if not lyrics.exists():
            raise typer.BadParameter(f"Lyrics file not found: {lyrics}")
        parsed_lyrics = parse_lrc(lyrics, audio.duration)
        if not parsed_lyrics:
            raise typer.BadParameter(f"No timed lines found in LRC file: {lyrics}")

    config = RenderConfig(
        width=width,
        height=height,
        fps=fps,
        sample_rate=sample_rate,
    )
    renderer = create_renderer(
        preset=preset,
        audio=audio,
        config=config,
        lyrics=parsed_lyrics,
        font_path=font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(np.ceil(audio.duration * fps)))
    print(
        f"Rendering {total_frames} frames at {fps} fps "
        f"({audio.duration:.2f}s) with preset '{preset}'.",
        file=sys.stderr,
    )

    ffmpeg_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(input_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output),
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_idx in range(total_frames):
            t = min(frame_idx / fps, audio.duration)
            frame = renderer.render_frame(t)
            process.stdin.write(frame.tobytes())

            if frame_idx == 0 or frame_idx % fps == 0:
                elapsed = frame_idx / fps
                print(
                    f"[render] {elapsed:7.2f}s / {audio.duration:7.2f}s",
                    file=sys.stderr,
                )
    except BrokenPipeError as exc:
        process.kill()
        raise RuntimeError("ffmpeg closed the frame pipe unexpectedly.") from exc
    finally:
        process.stdin.close()

    code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed during encode (exit code {code}).")
    print(f"Wrote video: {output}", file=sys.stderr)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
