from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LRC_TIMESTAMP = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class LyricWord:
    start: float
    end: float
    text: str


@dataclass
class LyricLine:
    start: float
    end: float
    text: str
    words: list[LyricWord] = field(default_factory=list)


def parse_lrc(path: Path, duration: float) -> list[LyricLine]:
    pending: list[tuple[float, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        timestamps = list(LRC_TIMESTAMP.finditer(raw))
        if not timestamps:
            continue

        text = LRC_TIMESTAMP.sub("", raw).strip()
        if not text:
            continue

        for match in timestamps:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            start = (minutes * 60.0) + seconds
            pending.append((start, text))

    pending.sort(key=lambda item: item[0])
    if not pending:
        return []

    lines: list[LyricLine] = []
    for index, (start, text) in enumerate(pending):
        if index + 1 < len(pending):
            end = pending[index + 1][0]
        else:
            end = duration
        if end <= start:
            end = start + 0.25
        lines.append(LyricLine(start=start, end=end, text=text))
    return lines


def active_lyric_index(lines: list[LyricLine], t: float) -> int | None:
    if not lines:
        return None
    for i, line in enumerate(lines):
        if line.start <= t < line.end:
            return i
    if t >= lines[-1].end:
        return len(lines) - 1
    return None
