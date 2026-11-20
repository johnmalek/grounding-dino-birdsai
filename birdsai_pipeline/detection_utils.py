from __future__ import annotations

import re


def normalize_phrase(value: object) -> str:
    """Normalize a model phrase for deterministic alias matching."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def map_phrase_to_category(
    phrase: object,
    aliases: dict[str, list[str]],
) -> str | None:
    """Map a phrase only when it matches exactly one final category."""
    normalized = normalize_phrase(phrase)
    matches = set()

    for category_name, values in aliases.items():
        for alias in sorted(values, key=len, reverse=True):
            normalized_alias = normalize_phrase(alias)
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized):
                matches.add(category_name)

    return next(iter(matches)) if len(matches) == 1 else None


def xyxy_to_xywh(
    box: list[float],
    width: int,
    height: int,
) -> tuple[list[float], list[float]] | None:
    """Clip an XYXY box and return COCO XYWH plus clipped XYXY."""
    x1, y1, x2, y2 = map(float, box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1], [x1, y1, x2, y2]


def axis_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return tile origins while ensuring the far edge is covered."""
    if length <= tile_size:
        return [0]

    overlap_pixels = int(round(tile_size * overlap))
    stride = max(1, tile_size - overlap_pixels)
    final_origin = length - tile_size
    origins = list(range(0, final_origin + 1, stride))
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def tile_windows(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Return PIL crop windows as left, top, right, bottom."""
    return [
        (
            left,
            top,
            min(left + tile_size, width),
            min(top + tile_size, height),
        )
        for top in axis_origins(height, tile_size, overlap)
        for left in axis_origins(width, tile_size, overlap)
    ]
