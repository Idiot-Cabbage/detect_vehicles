"""Zone loading and geometric helpers for parking violation detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import json

Point = Tuple[float, float]


@dataclass
class Zone:
    """Represents a no-parking zone polygon."""

    name: str
    points: List[Point]

    def as_tuple(self) -> Tuple[str, List[Point]]:
        return self.name, self.points


@dataclass
class ZoneConfig:
    """Holds the camera-specific no-parking zones."""

    image_width: int
    image_height: int
    zones: List[Zone]


class ZoneConfigError(ValueError):
    """Raised when the zone configuration file is invalid."""


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def load_zones(config_path: Path | str) -> ZoneConfig:
    """Load zone polygons from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration.

    Raises:
        ZoneConfigError: If required fields are missing or malformed.
    """

    path = Path(config_path)
    if not path.exists():
        raise ZoneConfigError(f"Zone config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for required_key in ("image_width", "image_height", "no_parking_zones"):
        if required_key not in data:
            raise ZoneConfigError(f"Missing '{required_key}' in zone config: {path}")

    zones: List[Zone] = []
    for idx, raw_zone in enumerate(data.get("no_parking_zones", [])):
        name = raw_zone.get("name")
        points = raw_zone.get("points")
        if not name or not isinstance(points, Iterable):
            raise ZoneConfigError(f"Zone entry #{idx} is invalid: {raw_zone}")
        points_list = _validate_points(points, idx)
        zones.append(Zone(name=name, points=points_list))

    return ZoneConfig(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        zones=zones,
    )


def _validate_points(points: Iterable[Sequence[float]], idx: int) -> List[Point]:
    point_list: List[Point] = []
    for p_idx, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ZoneConfigError(
                f"Zone entry #{idx} has an invalid point at index {p_idx}: {point}"
            )
        x, y = float(point[0]), float(point[1])
        point_list.append((x, y))
    if len(point_list) < 3:
        raise ZoneConfigError(f"Zone entry #{idx} must contain at least 3 points")
    return point_list


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray casting algorithm for point-in-polygon.

    Returns True if the point (x, y) lies inside or on the boundary of the polygon.
    """

    inside = False
    n = len(polygon)
    if n < 3:
        return inside

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        intersect = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersect:
            inside = not inside
    return inside


def compute_ioa(
    bbox: Sequence[float], polygon: Sequence[Point], *, require_shapely: bool = False
) -> float:
    """Compute the intersection-over-area (IOA) between a bbox and polygon.

    Args:
        bbox: Bounding box [x1, y1, x2, y2].
        polygon: Polygon points.
        require_shapely: If True, raise when shapely is missing; otherwise return 0.
    """

    try:
        from shapely.geometry import Polygon
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        if require_shapely:
            raise ZoneConfigError(
                "Shapely is required for IOA calculation but is not installed."
            )
        return 0.0

    x1, y1, x2, y2 = bbox
    box_poly = Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
    zone_poly = Polygon(polygon)
    if not box_poly.is_valid or not zone_poly.is_valid:
        return 0.0

    intersection_area = box_poly.intersection(zone_poly).area
    box_area = box_poly.area
    return intersection_area / box_area if box_area > 0 else 0.0
