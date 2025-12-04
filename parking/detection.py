"""Detection pipeline and visualization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from .zones import ZoneConfig, Zone, point_in_polygon, compute_ioa, VEHICLE_CLASSES


@dataclass
class Detection:
    bbox: List[float]
    cls: int
    label: str
    score: float
    is_violation: bool = False
    zone_names: List[str] | None = None

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, y2


@dataclass
class DetectionParams:
    conf_thres: float = 0.4
    iou_thres: float = 0.45
    use_ioa: bool = False
    ioa_thres: float = 0.3


class ParkingDetector:
    def __init__(self, model_path: str, device: str | None = None):
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

    def detect_vehicles(self, image_path: Path, params: DetectionParams) -> List[Detection]:
        results = self.model(
            str(image_path),
            conf=params.conf_thres,
            iou=params.iou_thres,
            classes=list(VEHICLE_CLASSES.keys()),
        )
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        cls=cls_id,
                        label=VEHICLE_CLASSES.get(cls_id, str(cls_id)),
                        score=score,
                    )
                )
        return detections

    def evaluate_violation(
        self, detection: Detection, zones: Iterable[Zone], params: DetectionParams
    ) -> Detection:
        bottom_center = detection.bottom_center
        is_inside = False
        matched_zones: List[str] = []
        for zone in zones:
            if point_in_polygon(bottom_center[0], bottom_center[1], zone.points):
                is_inside = True
                matched_zones.append(zone.name)
            if params.use_ioa:
                ioa = compute_ioa(detection.bbox, zone.points, require_shapely=True)
                if ioa >= params.ioa_thres:
                    is_inside = True
                    if zone.name not in matched_zones:
                        matched_zones.append(zone.name)
        detection.is_violation = is_inside
        detection.zone_names = matched_zones
        return detection

    def process_image(
        self, image_path: Path, zone_config: ZoneConfig, params: DetectionParams
    ) -> List[Detection]:
        detections = self.detect_vehicles(image_path, params)
        return [self.evaluate_violation(det, zone_config.zones, params) for det in detections]


def draw_results(
    image_path: Path,
    detections: Sequence[Detection],
    zones: Sequence[Zone],
    save_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    for zone in zones:
        pts = np.array(zone.points, dtype=np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
        cv2.putText(
            image,
            zone.name,
            (int(pts[0][0]), int(pts[0][1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        color = (0, 0, 255) if det.is_violation else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_parts = [det.label, f"{det.score:.2f}"]
        if det.is_violation:
            zones_text = "|".join(det.zone_names or [])
            label_parts.append(zones_text)
        label = " | ".join(label_parts)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def detections_to_rows(image_path: Path, detections: Sequence[Detection]) -> List[List[str]]:
    rows: List[List[str]] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        rows.append(
            [
                image_path.name,
                det.label,
                f"{det.score:.4f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
                "1" if det.is_violation else "0",
                "|".join(det.zone_names or []),
            ]
        )
    return rows
