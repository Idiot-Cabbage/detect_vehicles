"""Simple OpenCV polygon editor for creating no-parking zone configs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Reference image for drawing zones")
    parser.add_argument("--output", type=Path, default=Path("zones.json"), help="Output JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Failed to load image: {args.image}")

    clone = image.copy()
    zones: List[dict] = []
    current_points: List[Point] = []

    def redraw() -> None:
        canvas = clone.copy()
        for zone in zones:
            pts = zone["points"]
            if len(pts) >= 2:
                contour = cv2.UMat(np.array(pts, dtype=np.int32))
                cv2.polylines(canvas, [contour], True, (0, 165, 255), 2)
        for idx, pt in enumerate(current_points):
            cv2.circle(canvas, pt, 4, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                str(idx + 1),
                (pt[0] + 4, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.imshow("zone_editor", canvas)

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            redraw()

    cv2.namedWindow("zone_editor")
    cv2.setMouseCallback("zone_editor", on_click)
    redraw()
    print("Left click to add points. Press 'n' to finish a polygon, 'u' to undo, 'r' to reset current, 's' to save & exit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("u") and current_points:
            current_points.pop()
            redraw()
        elif key == ord("r"):
            current_points.clear()
            redraw()
        elif key == ord("n"):
            if len(current_points) < 3:
                print("Need at least 3 points to close a polygon")
                continue
            name = f"zone_{len(zones) + 1}"
            zones.append({"name": name, "points": current_points.copy()})
            current_points.clear()
            redraw()
            print(f"Added polygon '{name}'.")
        elif key == ord("s"):
            data = {
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "no_parking_zones": zones,
            }
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved zones to {args.output}")
            break
        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
