"""Command-line tool for YOLOv8-based no-parking detection on images."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from parking.detection import (
    Detection,
    DetectionParams,
    ParkingDetector,
    draw_results,
    detections_to_rows,
)
from parking.zones import ZoneConfig, load_zones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img_dir", required=True, type=Path, help="Directory of images")
    parser.add_argument("--zone_config", required=True, type=Path, help="Path to zones JSON")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Compute device, e.g. 'cpu' or 'cuda'")
    parser.add_argument("--output_dir", type=Path, default=Path("runs"), help="Output root")
    parser.add_argument(
        "--use_ioa",
        action="store_true",
        help="Enable IOA filtering (requires shapely)",
    )
    parser.add_argument("--ioa_thres", type=float, default=0.3, help="IOA threshold")
    parser.add_argument(
        "--recursive", action="store_true", help="Search for images in subdirectories"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["jpg", "jpeg", "png"],
        help="Image extensions to include",
    )
    return parser.parse_args()


def find_images(img_dir: Path, extensions: Iterable[str], recursive: bool) -> List[Path]:
    exts = {ext.lower().lstrip(".") for ext in extensions}
    pattern = "**/*" if recursive else "*"
    images = [p for p in img_dir.glob(pattern) if p.suffix.lower().lstrip(".") in exts]
    return sorted(images)


def save_csv(rows: List[List[str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "image_name",
        "cls",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "is_violation",
        "zone_names",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    zone_config: ZoneConfig = load_zones(args.zone_config)
    params = DetectionParams(
        conf_thres=args.conf, iou_thres=args.iou, use_ioa=args.use_ioa, ioa_thres=args.ioa_thres
    )

    detector = ParkingDetector(args.model, device=args.device)

    images = find_images(args.img_dir, args.formats, args.recursive)
    if not images:
        raise SystemExit(f"No images found in {args.img_dir}")

    vis_dir = args.output_dir / "vis"
    csv_path = args.output_dir / "results.csv"

    all_rows: List[List[str]] = []
    for image_path in images:
        detections: List[Detection] = detector.process_image(image_path, zone_config, params)
        draw_results(image_path, detections, zone_config.zones, vis_dir / image_path.name)
        all_rows.extend(detections_to_rows(image_path, detections))

    save_csv(all_rows, csv_path)
    print(f"Processed {len(images)} images. CSV saved to {csv_path}, visualizations in {vis_dir}.")


if __name__ == "__main__":
    main()
