from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .audio import AudioData, segment_around_time
from .lyrics import LyricLine, active_lyric_index

PRESET_OSCILLOSCOPE_3D = "oscilloscope-3d"
PRESET_KINETIC_LYRICS = "kinetic-lyrics"
PRESETS = (PRESET_OSCILLOSCOPE_3D, PRESET_KINETIC_LYRICS)


@dataclass
class RenderConfig:
    width: int
    height: int
    fps: int
    sample_rate: int
    primary_color: tuple[int, int, int] = (88, 238, 255)
    secondary_color: tuple[int, int, int] = (255, 170, 82)
    bg_top: tuple[int, int, int] = (9, 12, 28)
    bg_bottom: tuple[int, int, int] = (2, 3, 9)


def create_renderer(
    preset: str,
    audio: AudioData,
    config: RenderConfig,
    lyrics: list[LyricLine] | None = None,
    font_path: Path | None = None,
) -> "BaseRenderer":
    if preset == PRESET_OSCILLOSCOPE_3D:
        return Oscilloscope3DRenderer(audio=audio, config=config)
    if preset == PRESET_KINETIC_LYRICS:
        if not lyrics:
            raise ValueError("Preset 'kinetic-lyrics' requires a non-empty lyrics file.")
        return KineticLyricsRenderer(
            audio=audio,
            lyrics=lyrics,
            config=config,
            font_path=font_path,
        )
    raise ValueError(f"Unknown preset: {preset}")


class BaseRenderer:
    def render_frame(self, t: float) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Oscilloscope3DRenderer(BaseRenderer):
    audio: AudioData
    config: RenderConfig
    window_size: int = 4096
    points: int = 900
    trail_length: int = 16
    trail: deque[np.ndarray] = field(default_factory=deque)
    bg_cache: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.trail = deque(maxlen=self.trail_length)
        self.bg_cache = gradient_bg(
            self.config.width,
            self.config.height,
            self.config.bg_top,
            self.config.bg_bottom,
        )
        self.x_positions = np.linspace(
            72.0,
            self.config.width - 72.0,
            self.points,
            dtype=np.float32,
        )

    def render_frame(self, t: float) -> np.ndarray:
        wave = self._wave_points(t)
        energy = self._rms(wave)
        centroid = self._centroid_hz(t)
        self.trail.append(wave)

        assert self.bg_cache is not None
        image = Image.fromarray(self.bg_cache.copy(), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        hue = np.interp(centroid, [120.0, 7000.0], [0.56, 0.04])
        hue = float(np.clip(hue, 0.0, 1.0))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.74, 1.0)
        base_color = (int(r * 255), int(g * 255), int(b * 255))

        pulse = int(np.interp(energy, [0.0, 0.5], [42.0, 180.0]))
        center = (self.config.width // 2, int(self.config.height * 0.55))
        for radius in (220 + pulse, 330 + pulse // 2):
            draw.ellipse(
                (
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                ),
                outline=(base_color[0], base_color[1], base_color[2], 28),
                width=2,
            )

        amp = self.config.height * (0.24 + (0.42 * min(energy * 3.0, 1.0)))
        size = len(self.trail)
        for i, trail_wave in enumerate(self.trail):
            depth = (i + 1) / max(size, 1)
            shift_y = (1.0 - depth) * 95.0
            shift_x = (1.0 - depth) * 46.0
            alpha = int(35 + (190 * depth))
            thick = max(1, int(1 + (4 * depth)))

            y = (self.config.height * 0.56) - (trail_wave * amp * depth) - shift_y
            x = self.x_positions + shift_x
            points = [(float(px), float(py)) for px, py in zip(x, y)]

            glow_color = (base_color[0], base_color[1], base_color[2], alpha // 4)
            line_color = (base_color[0], base_color[1], base_color[2], alpha)

            draw.line(points, fill=glow_color, width=thick * 4)
            draw.line(points, fill=line_color, width=thick)

        horizon_y = int(self.config.height * 0.56)
        draw.line(
            [(36, horizon_y), (self.config.width - 36, horizon_y)],
            fill=(180, 208, 255, 45),
            width=1,
        )
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _wave_points(self, t: float) -> np.ndarray:
        chunk = segment_around_time(
            self.audio.samples,
            self.audio.sample_rate,
            t,
            self.window_size,
        )
        src = np.linspace(0, self.window_size - 1, self.window_size, dtype=np.float32)
        dst = np.linspace(0, self.window_size - 1, self.points, dtype=np.float32)
        interp = np.interp(dst, src, chunk)
        return interp.astype(np.float32)

    def _centroid_hz(self, t: float) -> float:
        chunk = segment_around_time(
            self.audio.samples,
            self.audio.sample_rate,
            t,
            self.window_size,
        )
        windowed = chunk * np.hanning(len(chunk))
        spec = np.abs(np.fft.rfft(windowed))
        if spec.sum() <= 1e-8:
            return 200.0
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.audio.sample_rate)
        return float(np.sum(freqs * spec) / np.sum(spec))

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))


@dataclass
class KineticLyricsRenderer(BaseRenderer):
    audio: AudioData
    lyrics: list[LyricLine]
    config: RenderConfig
    font_path: Path | None = None
    points: int = 700
    window_size: int = 4096
    bg_cache: np.ndarray | None = None
    prev_energy: float = 0.0

    def __post_init__(self) -> None:
        self.bg_cache = gradient_bg(
            self.config.width,
            self.config.height,
            (14, 8, 24),
            (2, 4, 11),
        )
        self.x_positions = np.linspace(
            68.0,
            self.config.width - 68.0,
            self.points,
            dtype=np.float32,
        )
        self.measure_canvas = Image.new("RGB", (32, 32))
        self.measure_draw = ImageDraw.Draw(self.measure_canvas)
        self.layout_cache: dict[str, list[list[tuple[str, int, int]]]] = {}
        self.font_main = load_font(74, self.font_path)
        self.font_small = load_font(40, self.font_path)

    def render_frame(self, t: float) -> np.ndarray:
        wave = self._wave_points(t)
        energy = self._rms(wave)
        beat = max(0.0, energy - self.prev_energy)
        self.prev_energy = (0.86 * self.prev_energy) + (0.14 * energy)

        bands = self._frequency_bands(t, 20)
        color_a = np.array(self.config.primary_color, dtype=np.float32)
        color_b = np.array(self.config.secondary_color, dtype=np.float32)

        assert self.bg_cache is not None
        image = Image.fromarray(self.bg_cache.copy(), mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        self._draw_side_bars(draw, bands, color_a, color_b)
        self._draw_floor_wave(draw, wave, energy, color_a)
        self._draw_lyrics(draw, t, beat)

        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _draw_side_bars(
        self,
        draw: ImageDraw.ImageDraw,
        bands: np.ndarray,
        color_a: np.ndarray,
        color_b: np.ndarray,
    ) -> None:
        h = self.config.height
        for i, val in enumerate(bands):
            strength = float(np.clip(val * 8.0, 0.0, 1.0))
            bar_h = int((h * 0.52) * strength)
            mix = i / max(len(bands) - 1, 1)
            color = ((1.0 - mix) * color_a) + (mix * color_b)
            fill = (int(color[0]), int(color[1]), int(color[2]), int(55 + 180 * strength))

            width = 8 + int(5 * strength)
            gap = 10
            x_left = 40 + i * (width + gap)
            x_right = self.config.width - x_left
            y0 = h - 34
            y1 = y0 - bar_h
            draw.rectangle((x_left, y1, x_left + width, y0), fill=fill)
            draw.rectangle((x_right - width, y1, x_right, y0), fill=fill)

    def _draw_floor_wave(
        self,
        draw: ImageDraw.ImageDraw,
        wave: np.ndarray,
        energy: float,
        color: np.ndarray,
    ) -> None:
        amp = self.config.height * (0.08 + 0.2 * min(energy * 3.5, 1.0))
        y = (self.config.height * 0.74) - (wave * amp)
        points = [(float(px), float(py)) for px, py in zip(self.x_positions, y)]
        alpha = int(np.interp(energy, [0.0, 0.5], [90.0, 240.0]))
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        draw.line(points, fill=(rgb[0], rgb[1], rgb[2], alpha // 3), width=14)
        draw.line(points, fill=(rgb[0], rgb[1], rgb[2], alpha), width=4)

    def _draw_lyrics(self, draw: ImageDraw.ImageDraw, t: float, beat: float) -> None:
        index = active_lyric_index(self.lyrics, t)
        if index is None:
            return

        line = self.lyrics[index]
        words = line.text.split()
        if not words:
            return

        duration = max(line.end - line.start, 0.25)
        progress = float(np.clip((t - line.start) / duration, 0.0, 1.0))
        highlight = progress * len(words)
        active_word = min(len(words) - 1, int(highlight))
        active_fraction = highlight - active_word

        layout = self._layout_words(line.text)
        line_height = int(getattr(self.font_main, "size", 48) * 1.3)
        y_base = int(self.config.height * 0.34)

        for row_idx, row in enumerate(layout):
            total_w = 0
            for j, (_, _, width) in enumerate(row):
                total_w += width
                if j != len(row) - 1:
                    total_w += self.space_w

            x = (self.config.width - total_w) // 2
            y = y_base + (row_idx * line_height)
            for word, word_idx, width in row:
                rgba = (214, 222, 238, 185)
                wobble_y = 0

                if word_idx < active_word:
                    rgba = (255, 247, 201, 255)
                elif word_idx == active_word:
                    pulse = np.sin(active_fraction * np.pi)
                    intensity = int(210 + 45 * pulse + (95 * min(beat * 8.0, 1.0)))
                    rgba = (255, intensity, 180, 255)
                    wobble_y = -int(12 * pulse)

                draw.text((x + 3, y + 4 + wobble_y), word, font=self.font_main, fill=(0, 0, 0, 140))
                draw.text((x, y + wobble_y), word, font=self.font_main, fill=rgba)
                x += width + self.space_w

        if index + 1 < len(self.lyrics):
            next_line = self.lyrics[index + 1].text
            bbox = draw.textbbox((0, 0), next_line, font=self.font_small)
            w = bbox[2] - bbox[0]
            x = (self.config.width - w) // 2
            y = int(self.config.height * 0.86)
            draw.text((x, y), next_line, font=self.font_small, fill=(188, 198, 215, 145))

    def _layout_words(self, text: str) -> list[list[tuple[str, int, int]]]:
        cached = self.layout_cache.get(text)
        if cached is not None:
            return cached

        words = text.split()
        max_width = int(self.config.width * 0.86)
        self.space_w = self._text_width(" ", self.font_main)

        rows: list[list[tuple[str, int, int]]] = []
        row: list[tuple[str, int, int]] = []
        row_w = 0

        for idx, word in enumerate(words):
            word_w = self._text_width(word, self.font_main)
            required = word_w if not row else row_w + self.space_w + word_w
            if row and required > max_width:
                rows.append(row)
                row = [(word, idx, word_w)]
                row_w = word_w
            else:
                if row:
                    row_w += self.space_w + word_w
                else:
                    row_w = word_w
                row.append((word, idx, word_w))

        if row:
            rows.append(row)

        self.layout_cache[text] = rows
        return rows

    def _text_width(self, text: str, font: ImageFont.ImageFont) -> int:
        bbox = self.measure_draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])

    def _wave_points(self, t: float) -> np.ndarray:
        chunk = segment_around_time(
            self.audio.samples,
            self.audio.sample_rate,
            t,
            self.window_size,
        )
        src = np.linspace(0, self.window_size - 1, self.window_size, dtype=np.float32)
        dst = np.linspace(0, self.window_size - 1, self.points, dtype=np.float32)
        return np.interp(dst, src, chunk).astype(np.float32)

    def _frequency_bands(self, t: float, bands: int) -> np.ndarray:
        chunk = segment_around_time(
            self.audio.samples,
            self.audio.sample_rate,
            t,
            self.window_size,
        )
        windowed = chunk * np.hanning(len(chunk))
        spectrum = np.abs(np.fft.rfft(windowed))
        if np.max(spectrum) <= 1e-8:
            return np.zeros(bands, dtype=np.float32)

        bins = np.array_split(spectrum, bands)
        values = np.array([float(np.mean(part)) for part in bins], dtype=np.float32)
        values /= float(np.max(values))
        return values

    @staticmethod
    def _rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))


def gradient_bg(
    width: int,
    height: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    base = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        base[..., i] = ((1.0 - y) * top[i]) + (y * bottom[i])

    vignette = np.clip(1.0 - (0.8 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)), 0.3, 1.0)
    base *= vignette[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def load_font(size: int, explicit: Path | None) -> ImageFont.ImageFont:
    candidates = []
    if explicit:
        candidates.append(str(explicit))
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Futura.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()
