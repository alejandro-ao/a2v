from __future__ import annotations

from pathlib import Path

from .lyrics import LyricLine, LyricWord


def transcribe_with_whisper(
    audio_path: Path,
    model_name: str = "small",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "default",
) -> list[LyricLine]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'faster-whisper'. Run `uv sync` to install it."
        ) from exc

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    lines: list[LyricLine] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        start = float(segment.start)
        end = float(segment.end)
        if end <= start:
            end = start + 0.2

        words: list[LyricWord] = []
        for word in segment.words or []:
            word_text = (word.word or "").strip()
            if not word_text:
                continue

            word_start = float(word.start) if word.start is not None else start
            word_end = float(word.end) if word.end is not None else min(end, word_start + 0.2)
            if word_end <= word_start:
                word_end = word_start + 0.05
            words.append(LyricWord(start=word_start, end=word_end, text=word_text))

        lines.append(LyricLine(start=start, end=end, text=text, words=words))

    return _normalize_lines(lines)


def _normalize_lines(lines: list[LyricLine]) -> list[LyricLine]:
    if not lines:
        return lines

    lines.sort(key=lambda line: line.start)
    for index, line in enumerate(lines):
        if index + 1 < len(lines):
            next_start = lines[index + 1].start
            if line.end > next_start:
                line.end = max(line.start + 0.1, next_start)
        if line.end <= line.start:
            line.end = line.start + 0.1
    return lines

