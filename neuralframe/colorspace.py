"""Colour-space definitions: YCbCr matrices, primaries, transfer functions, ranges and chroma siting.

Names follow FFmpeg/ffprobe (H.273) spellings so probe output maps directly and encoder tags round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt


class Matrix(StrEnum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020NC = "bt2020nc"
    SMPTE240M = "smpte240m"
    FCC = "fcc"
    GBR = "gbr"


class Primaries(StrEnum):
    BT709 = "bt709"
    BT470BG = "bt470bg"
    SMPTE170M = "smpte170m"
    SMPTE240M = "smpte240m"
    BT2020 = "bt2020"
    BT470M = "bt470m"


class Transfer(StrEnum):
    BT709 = "bt709"
    SMPTE170M = "smpte170m"
    BT2020_10 = "bt2020-10"
    BT2020_12 = "bt2020-12"
    SMPTE2084 = "smpte2084"
    ARIB_STD_B67 = "arib-std-b67"
    IEC61966_2_1 = "iec61966-2-1"
    GAMMA22 = "gamma22"
    GAMMA28 = "gamma28"
    SMPTE240M = "smpte240m"
    LINEAR = "linear"


class ColorRange(StrEnum):
    LIMITED = "tv"
    FULL = "pc"


class ChromaLocation(StrEnum):
    LEFT = "left"
    CENTER = "center"
    TOPLEFT = "topleft"
    TOP = "top"
    BOTTOMLEFT = "bottomleft"
    BOTTOM = "bottom"


HDR_TRANSFERS = frozenset({Transfer.SMPTE2084, Transfer.ARIB_STD_B67})

# Luma coefficients (Kr, Kb); Kg = 1 - Kr - Kb.
KR_KB: dict[Matrix, tuple[float, float]] = {
    Matrix.BT601: (0.299, 0.114),
    Matrix.BT709: (0.2126, 0.0722),
    Matrix.BT2020NC: (0.2627, 0.0593),
    Matrix.SMPTE240M: (0.212, 0.087),
    Matrix.FCC: (0.30, 0.11),
}

_D65 = (0.3127, 0.3290)
_ILLUMINANT_C = (0.310, 0.316)

# (red xy, green xy, blue xy, white xy)
PRIMARIES_XY: dict[Primaries, tuple[tuple[float, float], ...]] = {
    Primaries.BT709: ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), _D65),
    Primaries.BT470BG: ((0.640, 0.330), (0.290, 0.600), (0.150, 0.060), _D65),
    Primaries.SMPTE170M: ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), _D65),
    Primaries.SMPTE240M: ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), _D65),
    Primaries.BT2020: ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), _D65),
    Primaries.BT470M: ((0.670, 0.330), (0.210, 0.710), (0.140, 0.080), _ILLUMINANT_C),
}

_MATRIX_ALIASES: dict[str, Matrix] = {
    "bt709": Matrix.BT709,
    "smpte170m": Matrix.BT601,
    "bt470bg": Matrix.BT601,
    "bt601": Matrix.BT601,
    "bt2020nc": Matrix.BT2020NC,
    "bt2020c": Matrix.BT2020NC,
    "bt2020_ncl": Matrix.BT2020NC,
    "smpte240m": Matrix.SMPTE240M,
    "fcc": Matrix.FCC,
    "gbr": Matrix.GBR,
    "rgb": Matrix.GBR,
}


def parse_matrix(name: str | None) -> Matrix | None:
    return _MATRIX_ALIASES.get((name or "").lower())


def parse_primaries(name: str | None) -> Primaries | None:
    try:
        return Primaries((name or "").lower())
    except ValueError:
        return None


def parse_transfer(name: str | None) -> Transfer | None:
    value = (name or "").lower()
    aliases = {"bt470m": Transfer.GAMMA22, "bt470bg": Transfer.GAMMA28, "srgb": Transfer.IEC61966_2_1}
    if value in aliases:
        return aliases[value]
    try:
        return Transfer(value)
    except ValueError:
        return None


def parse_range(name: str | None) -> ColorRange | None:
    value = (name or "").lower()
    if value in ("tv", "mpeg", "limited"):
        return ColorRange.LIMITED
    if value in ("pc", "jpeg", "full"):
        return ColorRange.FULL
    return None


def parse_chroma_location(name: str | None) -> ChromaLocation | None:
    try:
        return ChromaLocation((name or "").lower())
    except ValueError:
        return None


def chroma_siting(location: ChromaLocation) -> tuple[bool, bool]:
    """(horizontally co-sited, vertically co-sited) for a chroma location; False means centred."""
    horizontal = location in (ChromaLocation.LEFT, ChromaLocation.TOPLEFT, ChromaLocation.BOTTOMLEFT)
    vertical = location in (ChromaLocation.TOPLEFT, ChromaLocation.TOP)
    return horizontal, vertical


def ffmpeg_colorspace_tag(matrix: Matrix, primaries: Primaries) -> str:
    """FFmpeg `-colorspace` value; BT.601 is spelled by its system (525 vs 625 lines)."""
    if matrix is Matrix.BT601:
        return "bt470bg" if primaries is Primaries.BT470BG else "smpte170m"
    return matrix.value


def _xy_to_xyz(x: float, y: float) -> npt.NDArray[np.float64]:
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz(primaries: Primaries) -> npt.NDArray[np.float64]:
    r, g, b, w = PRIMARIES_XY[primaries]
    m = np.stack([_xy_to_xyz(*r), _xy_to_xyz(*g), _xy_to_xyz(*b)], axis=1)
    scale = np.linalg.solve(m, _xy_to_xyz(*w))
    return m * scale[np.newaxis, :]


_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]], dtype=np.float64
)


def gamut_matrix(src: Primaries, dst: Primaries) -> npt.NDArray[np.float32]:
    """Linear-light RGB(src) → RGB(dst) matrix, with Bradford adaptation if white points differ."""
    m = np.linalg.inv(rgb_to_xyz(dst)) @ _chromatic_adaptation(src, dst) @ rgb_to_xyz(src)
    return m.astype(np.float32)


def _chromatic_adaptation(src: Primaries, dst: Primaries) -> npt.NDArray[np.float64]:
    ws, wd = PRIMARIES_XY[src][3], PRIMARIES_XY[dst][3]
    if ws == wd:
        return np.eye(3, dtype=np.float64)
    cone_s = _BRADFORD @ _xy_to_xyz(*ws)
    cone_d = _BRADFORD @ _xy_to_xyz(*wd)
    return np.asarray(np.linalg.inv(_BRADFORD) @ np.diag(cone_d / cone_s) @ _BRADFORD, dtype=np.float64)


def same_gamut(a: Primaries, b: Primaries) -> bool:
    return PRIMARIES_XY[a] == PRIMARIES_XY[b]


@dataclass(frozen=True)
class ColorPlan:
    """Resolved colour handling for one job: how to read the source and how to write/tag the output."""

    in_matrix: Matrix
    in_range: ColorRange
    in_primaries: Primaries
    in_transfer: Transfer
    in_chroma: ChromaLocation
    out_matrix: Matrix
    out_range: ColorRange
    out_primaries: Primaries
    out_transfer: Transfer
    out_chroma: ChromaLocation = ChromaLocation.LEFT
    out_bit_depth: int = 8
    convert_gamut: bool = False
    hdr: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def gamut(self) -> npt.NDArray[np.float32] | None:
        return gamut_matrix(self.in_primaries, self.out_primaries) if self.convert_gamut else None

    def ffmpeg_tags(self) -> dict[str, str]:
        """Values for FFmpeg's -colorspace/-color_primaries/-color_trc/-color_range/-chroma_sample_location."""
        return {
            "colorspace": ffmpeg_colorspace_tag(self.out_matrix, self.out_primaries),
            "color_primaries": self.out_primaries.value,
            "color_trc": self.out_transfer.value,
            "color_range": self.out_range.value,
            "chroma_sample_location": self.out_chroma.value,
        }

    def summary(self) -> dict[str, object]:
        return {
            "input": {
                "matrix": self.in_matrix.value,
                "range": self.in_range.value,
                "primaries": self.in_primaries.value,
                "transfer": self.in_transfer.value,
                "chroma_location": self.in_chroma.value,
            },
            "output": {**self.ffmpeg_tags(), "bit_depth": self.out_bit_depth},
            "convert_gamut": self.convert_gamut,
            "hdr_passthrough": self.hdr,
            "notes": list(self.notes),
        }
