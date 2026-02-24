from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples)) / float(self.sample_rate)


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(
            f"Missing required binary '{name}'. Install it first and make sure it is in PATH."
        )
    return resolved


def decode_audio_mono(path: Path, sample_rate: int = 22_050) -> AudioData:
    ffmpeg = require_binary("ffmpeg")
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr or 'unknown error'}")

    raw = np.frombuffer(result.stdout, dtype=np.int16)
    samples = raw.astype(np.float32) / 32768.0
    if samples.size == 0:
        raise RuntimeError("Decoded audio has zero samples.")
    return AudioData(samples=samples, sample_rate=sample_rate)


def segment_around_time(
    samples: np.ndarray,
    sample_rate: int,
    center_t: float,
    window_size: int,
) -> np.ndarray:
    center_index = int(center_t * sample_rate)
    half = window_size // 2
    start = center_index - half
    end = start + window_size

    left_pad = max(0, -start)
    right_pad = max(0, end - len(samples))

    clipped_start = max(0, start)
    clipped_end = min(len(samples), end)
    chunk = samples[clipped_start:clipped_end]

    if left_pad or right_pad:
        chunk = np.pad(chunk, (left_pad, right_pad))

    if len(chunk) != window_size:
        chunk = np.resize(chunk, window_size)

    return chunk

