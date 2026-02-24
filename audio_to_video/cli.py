from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from .audio import decode_audio_mono, require_binary
from .lyrics import parse_lrc
from .renderers import PRESET_KINETIC_LYRICS, PRESETS, RenderConfig, create_renderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a2v",
        description="Render audio-reactive MP4 videos from an audio file.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    presets_cmd = sub.add_parser("presets", help="List available visual presets.")
    presets_cmd.set_defaults(func=cmd_presets)

    render_cmd = sub.add_parser("render", help="Render an MP4 video from an audio file.")
    render_cmd.add_argument("input_audio", type=Path, help="Input audio path (mp3/wav/m4a/...).")
    render_cmd.add_argument("-o", "--output", type=Path, required=True, help="Output MP4 path.")
    render_cmd.add_argument(
        "-p",
        "--preset",
        choices=PRESETS,
        default=PRESETS[0],
        help="Visual preset to render.",
    )
    render_cmd.add_argument(
        "--lyrics",
        type=Path,
        default=None,
        help="Path to .lrc lyrics file (required for kinetic-lyrics).",
    )
    render_cmd.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional path to a TTF/OTF/TTC font for lyric rendering.",
    )
    render_cmd.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    render_cmd.add_argument("--width", type=int, default=1920, help="Video width (default: 1920).")
    render_cmd.add_argument("--height", type=int, default=1080, help="Video height (default: 1080).")
    render_cmd.add_argument(
        "--sample-rate",
        type=int,
        default=22_050,
        help="Audio sample rate used for analysis (default: 22050).",
    )
    render_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    render_cmd.set_defaults(func=cmd_render)

    return parser


def cmd_presets(_args: argparse.Namespace) -> int:
    print("Available presets:")
    for preset in PRESETS:
        print(f"- {preset}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    ffmpeg = require_binary("ffmpeg")

    input_audio: Path = args.input_audio
    output: Path = args.output
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio does not exist: {input_audio}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
    if args.width < 64 or args.height < 64:
        raise ValueError("Width and height must both be >= 64.")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1.")

    print(f"Decoding audio: {input_audio}", file=sys.stderr)
    audio = decode_audio_mono(input_audio, sample_rate=args.sample_rate)

    lyrics = None
    if args.preset == PRESET_KINETIC_LYRICS:
        if args.lyrics is None:
            raise ValueError("--lyrics is required for preset 'kinetic-lyrics'.")
        if not args.lyrics.exists():
            raise FileNotFoundError(f"Lyrics file not found: {args.lyrics}")
        lyrics = parse_lrc(args.lyrics, audio.duration)
        if not lyrics:
            raise ValueError(f"No timed lines found in LRC file: {args.lyrics}")

    config = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
    )
    renderer = create_renderer(
        preset=args.preset,
        audio=audio,
        config=config,
        lyrics=lyrics,
        font_path=args.font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(np.ceil(audio.duration * args.fps)))
    print(
        f"Rendering {total_frames} frames at {args.fps} fps "
        f"({audio.duration:.2f}s) with preset '{args.preset}'.",
        file=sys.stderr,
    )

    ffmpeg_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
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
            t = min(frame_idx / args.fps, audio.duration)
            frame = renderer.render_frame(t)
            process.stdin.write(frame.tobytes())

            if frame_idx == 0 or frame_idx % args.fps == 0:
                elapsed = frame_idx / args.fps
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
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

