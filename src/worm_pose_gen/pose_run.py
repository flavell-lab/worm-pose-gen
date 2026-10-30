"""Shared pieces of a recording-level pose run.

Mask cleanup, tube rendering from stored poses, and the two drawings every
run produces: the overlay (tube boundary, centerline, head square and tail
circle) and the residual image, in which mask pixels the tube misses are
tinted blue and tube pixels outside the mask red.  ``scripts/fit_recording.py``
uses these while fitting; ``scripts/render_pose_run.py`` and
``scripts/compare_pose_runs.py`` use them on stored ``poses.npz`` arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
import torch

from .connected_components import largest_component
from .flat_field import apply_flat_field
from .mask_fit import fill_narrow_holes, render_tube_segments


BoolArray = NDArray[np.bool_]

TUBE_RGB = (90, 220, 140)
LINE_RGB = (255, 80, 165)
HEAD_RGB = (255, 210, 60)
TAIL_RGB = (80, 200, 255)
# Residual tints, blended half and half with the image.
MISSED_RGB = (0, 0, 127)  # mask the tube does not cover
EXTRA_RGB = (127, 0, 0)  # tube outside the mask
TEXT_RGB = (255, 255, 255)


def clean_mask(
    probability: np.ndarray,
    threshold: float,
    hole_radius: int,
    device: torch.device | str | None,
    *,
    fill_holes: bool = True,
) -> tuple[BoolArray, dict[str, int]]:
    """Threshold, then optionally fill the narrow holes; every component is kept.

    Returns the mask and its statistics.  The statistics (pixels a hole fill
    would add, component count, pixels outside the largest component) are
    computed whether or not the hole fill is applied, so the ambiguity flags
    ``holes`` and ``fragments`` keep their meaning on a raw mask;
    ``worm_pixels`` counts the mask that is returned.
    """

    raw = probability >= threshold
    stats = {"raw_worm_pixels": int(raw.sum()), "pixels_filled": 0, "components": 0, "pixels_outside_largest": 0, "worm_pixels": 0}
    if not raw.any():
        return raw, stats
    filled, added = fill_narrow_holes(raw, hole_radius, device=device)
    _, area, count = largest_component(filled)
    mask = filled if fill_holes else raw
    stats.update(
        pixels_filled=int(added), components=int(count), pixels_outside_largest=int(filled.sum()) - int(area), worm_pixels=int(mask.sum())
    )
    return mask, stats


def cleanup_options(summary: dict[str, Any]) -> dict[str, Any]:
    """``clean_mask`` keyword arguments recorded in a run's ``summary.json``."""

    cleanup = summary.get("mask_cleanup") or {}
    return {"hole_radius": int(cleanup.get("fill_holes_radius_px", 8)), "fill_holes": bool(cleanup.get("fill_holes", True))}


def render_tube(
    centerline_xy: NDArray[np.generic],
    width_profile: NDArray[np.generic],
    height: int,
    width: int,
    *,
    window: tuple[int, int, int, int] | None = None,
    margin: int = 48,
    device: torch.device | str | None = None,
    threshold: float = 0.5,
) -> BoolArray:
    """Hard tube occupancy of one stored pose on the full image.

    ``window`` is ``(x0, x1, y0, y1)``, typically the fit's crop; rendering is
    restricted to it grown by ``margin`` pixels, which is where the tube can be.
    """

    resolved = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if window is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0 = max(0, int(window[0]) - margin), max(0, int(window[2]) - margin)
        x1, y1 = min(width, int(window[1]) + margin), min(height, int(window[3]) + margin)
    curve = torch.as_tensor(np.asarray(centerline_xy, dtype=np.float64) - (x0, y0), dtype=torch.float32, device=resolved)[None]
    profile = torch.as_tensor(np.asarray(width_profile, dtype=np.float64), dtype=torch.float32, device=resolved)[None]
    with torch.no_grad():
        rendered = render_tube_segments(curve, profile, y1 - y0, x1 - x0)
    tube = np.zeros((height, width), dtype=bool)
    tube[y0:y1, x0:x1] = (rendered[0] >= threshold).cpu().numpy()
    return tube


def touches_border(mask: NDArray[np.generic], margin: int = 1) -> bool:
    """Whether any mask pixel lies within ``margin`` pixels of the image border.

    A body cut off by the camera edge keeps every centerline point inside the
    image, so the in-view count cannot tell it from a whole body; the mask
    reaching the border can.
    """

    binary = np.asarray(mask, dtype=bool)
    m = max(1, margin)
    return bool(binary[:m].any() or binary[-m:].any() or binary[:, :m].any() or binary[:, -m:].any())


def boundary(mask: NDArray[np.generic]) -> BoolArray:
    """Mask pixels with a 4-neighbor outside the mask."""

    binary = np.asarray(mask, dtype=bool)
    inner = binary.copy()
    inner[1:] &= binary[:-1]
    inner[:-1] &= binary[1:]
    inner[:, 1:] &= binary[:, :-1]
    inner[:, :-1] &= binary[:, 1:]
    return binary & ~inner


def tube_area_px(width_profile: NDArray[np.generic], body_length_px: float) -> float:
    """Area of a tube with this width profile along a body of this length."""

    profile = np.asarray(width_profile, dtype=np.float64)
    return float(np.trapezoid(profile) * body_length_px / max(len(profile) - 1, 1))


def _draw_pose(draw: ImageDraw.ImageDraw, centerline_xy: NDArray[np.generic], offset: tuple[float, float] = (0.0, 0.0)) -> None:
    points = [(float(x) - offset[0], float(y) - offset[1]) for x, y in centerline_xy]
    draw.line(points, fill=LINE_RGB, width=2)
    # Head is a square, tail (the thinner end, placed last) a circle.
    x, y = points[0]
    draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline=HEAD_RGB, width=2)
    x, y = points[-1]
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=TAIL_RGB, width=2)


def _resize(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    size = (max(2, int(round(image.width * scale)) // 2 * 2), max(2, int(round(image.height * scale)) // 2 * 2))
    return image.resize(size, Image.BILINEAR)


def draw_overlay(
    frame: np.ndarray,
    centerline_xy: NDArray[np.generic] | None,
    tube: NDArray[np.generic] | None,
    caption: str,
    scale: float = 1.0,
) -> np.ndarray:
    """Video frame: tube boundary, centerline with head/tail markers, caption."""

    rgb = np.repeat(np.asarray(frame, dtype=np.uint8)[:, :, None], 3, axis=2)
    if tube is not None:
        rgb[boundary(tube)] = TUBE_RGB
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if centerline_xy is not None:
        _draw_pose(draw, centerline_xy)
    draw.text((8, 8), caption, fill=TEXT_RGB)
    return np.asarray(_resize(image, scale))


def draw_residual(
    frame: np.ndarray,
    mask: NDArray[np.generic],
    tube: NDArray[np.generic],
    centerline_xy: NDArray[np.generic] | None,
    caption: str,
    *,
    window: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """Residual image: blue where the mask is not covered, red where the tube leaves the mask."""

    binary = np.asarray(mask, dtype=bool)
    occupied = np.asarray(tube, dtype=bool)
    rgb = np.repeat(np.asarray(frame, dtype=np.uint8)[:, :, None], 3, axis=2).astype(np.float32)
    missed = binary & ~occupied
    extra = occupied & ~binary
    rgb[missed] = 0.5 * rgb[missed] + np.asarray(MISSED_RGB, dtype=np.float32)
    rgb[extra] = 0.5 * rgb[extra] + np.asarray(EXTRA_RGB, dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    offset = (0.0, 0.0)
    if window is not None:
        x0, x1, y0, y1 = (int(v) for v in window)
        rgb = rgb[y0:y1, x0:x1]
        offset = (float(x0), float(y0))
    image = Image.fromarray(np.ascontiguousarray(rgb))
    draw = ImageDraw.Draw(image)
    if centerline_xy is not None:
        _draw_pose(draw, centerline_xy, offset)
    draw.text((8, 8), caption, fill=TEXT_RGB)
    return image


def residual_caption(frame_index: int, arrays: dict[str, np.ndarray], row: int, mask: NDArray[np.generic]) -> str:
    """One-line description of a stored fit and its mask agreement."""

    area_ratio = float(np.asarray(mask, dtype=bool).sum()) / max(tube_area_px(arrays["width_profile"][row], float(arrays["body_length_px"][row])), 1.0)
    parts = [
        f"frame {frame_index}",
        f"iou {float(arrays['iou'][row]):.3f}",
    ]
    if "ambiguity_score" in arrays and int(arrays["ambiguity_score"][row]):
        parts.append("flags " + ",".join(n for n in ("low_iou", "area_deficit", "area_excess", "self_contact", "holes", "fragments", "length_deviation", "pose_jump") if arrays[f"flag_{n}"][row]))
    parts += [
        f"len {float(arrays['body_length_px'][row]):.0f}",
        f"in-view {float(arrays['points_in_fov'][row]) / arrays['centerline_xy'].shape[1]:.2f}",
        f"mask/tube-area {area_ratio:.2f}",
    ]
    if "taper_asymmetry" in arrays and np.isfinite(arrays["taper_asymmetry"][row]):
        parts.append(f"taper {float(arrays['taper_asymmetry'][row]):+.2f}")
    if "best_start" in arrays:
        parts.append(f"best {arrays['best_start'][row]}")
    return "  ".join(parts)


def residual_rows(arrays: dict[str, np.ndarray], worst: int, requested: list[int]) -> list[int]:
    """Rows to dump: the ``worst`` lowest-IoU fitted frames plus the requested frame indices."""

    fitted = np.nonzero(arrays["fitted"])[0]
    rows: list[int] = []
    if worst > 0 and len(fitted):
        order = fitted[np.argsort(arrays["iou"][fitted])]
        rows.extend(int(r) for r in order[:worst])
    frame_index = arrays["frame_index"]
    for frame in requested:
        hits = np.nonzero(frame_index == frame)[0]
        if len(hits) and arrays["fitted"][hits[0]]:
            rows.append(int(hits[0]))
    return sorted(set(rows))


def flat_fielded(raw: NDArray[np.generic], field: Any) -> NDArray[np.uint8]:
    """The flat-fielded 8-bit frame; the raw frame when no field is given."""

    if field is None:
        return np.asarray(raw, dtype=np.uint8)
    return np.clip(np.rint(apply_flat_field(raw, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8)


SOURCE_LABELS = {1: "forward", 2: "backward"}


def overlay_caption(prefix: str, arrays: dict[str, np.ndarray], row: int) -> str:
    """One-line caption for a video frame from the stored arrays."""

    caption = f"{prefix} frame {int(arrays['frame_index'][row])}"
    if not arrays["fitted"][row]:
        return caption + "  no fit"
    n_points = arrays["centerline_xy"].shape[1]
    caption += (
        f"  iou {float(arrays['iou'][row]):.3f}  length {float(arrays['body_length_px'][row]):.0f} px"
        f"  width {float(arrays['width_px'][row]):.1f} px  in view {float(arrays['points_in_fov'][row]) / n_points:.2f}"
    )
    if "taper_asymmetry" in arrays and np.isfinite(arrays["taper_asymmetry"][row]):
        caption += f"  taper {float(arrays['taper_asymmetry'][row]):+.2f}"
    if "orientation_gap" in arrays and np.isfinite(arrays["orientation_gap"][row]):
        caption += f"  gap {float(arrays['orientation_gap'][row]):.3f}"
    if "source" in arrays and int(arrays["source"][row]) in SOURCE_LABELS:
        caption += f"  {SOURCE_LABELS[int(arrays['source'][row])]}"
    if "ambiguity_score" in arrays and int(arrays["ambiguity_score"][row]):
        caption += f"  score {int(arrays['ambiguity_score'][row])}"
    return caption


def write_overlay_video(
    path: Any,
    dataset: Any,
    field: Any,
    arrays: dict[str, np.ndarray],
    *,
    caption_prefix: str,
    fps: float = 20.0,
    scale: float = 1.0,
    quality: int = 5,
    slab: int = 64,
    device: torch.device | str | None = None,
) -> None:
    """Write the overlay MP4 from a run's final arrays.

    Frames are re-read from ``dataset`` (anything indexable by frame and frame
    range: a frame reader, an HDF5 dataset or an array) so the video always shows the poses as stored,
    including frames replaced by propagation after the independent pass.
    """

    import imageio.v2 as imageio

    frame_index = np.asarray(arrays["frame_index"])
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=quality, macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    try:
        for slab_start in range(0, len(frame_index), max(1, slab)):
            rows = range(slab_start, min(len(frame_index), slab_start + max(1, slab)))
            first, last = int(frame_index[rows[0]]), int(frame_index[rows[-1]])
            if last - first + 1 == len(rows):
                raw = np.asarray(dataset[first : last + 1], dtype=np.uint8)
            else:
                raw = np.stack([np.asarray(dataset[int(frame_index[r])], dtype=np.uint8) for r in rows])
            for row, raw_frame in zip(rows, raw, strict=True):
                frame = flat_fielded(raw_frame, field)
                tube = centerline = None
                if arrays["fitted"][row]:
                    centerline = arrays["centerline_xy"][row]
                    tube = render_tube(
                        centerline, arrays["width_profile"][row], *frame.shape, window=tuple(arrays["crop"][row]), device=device
                    )
                writer.append_data(draw_overlay(frame, centerline, tube, overlay_caption(caption_prefix, arrays, row), scale))
    finally:
        writer.close()


def run_label(summary: dict[str, Any]) -> str:
    """Short label of a run's width model for figure captions."""

    model = summary.get("width_model")
    if not model or not model.get("coefficients"):
        return "symmetric template"
    return f"{model['coefficients']} width coefficients"
