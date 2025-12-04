# YOLOv8 Parking Violation Detector

A lightweight offline tool to flag vehicles that occupy pre-defined **no-parking zones** in still images from a fixed camera. It uses official Ultralytics YOLOv8 models (e.g., `yolov8n.pt`, `yolov8s.pt`) and outputs both visualization overlays and structured CSV results.

## Features
- Detects COCO vehicle classes (car, motorcycle, bus, truck) using YOLOv8.
- Supports multiple polygonal no-parking zones per camera via JSON configuration.
- Violation check via bottom-center point-in-polygon, with optional IOA thresholding (Shapely).
- Exports annotated images (green = normal, red = violation) and CSV rows per detection.
- Includes an OpenCV helper (`zone_editor.py`) to click-and-save polygon configs.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing zone configuration
Use the included helper to draw polygons on a reference image from the camera:
```bash
python zone_editor.py path/to/reference.jpg --output zones_cam1.json
```
Controls:
- **Left click**: add a vertex
- **n**: close current polygon and store it
- **u**: undo last vertex
- **r**: reset the in-progress polygon
- **s**: save and exit (writes `image_width`, `image_height`, and `no_parking_zones`)

Example JSON schema:
```json
{
  "image_width": 4000,
  "image_height": 3000,
  "no_parking_zones": [
    {"name": "crosswalk_1", "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]},
    {"name": "intersection_area", "points": [[...], [...], [...]]}
  ]
}
```

## Running detection
```bash
python detect_parking.py \
  --img_dir ./images \
  --zone_config ./zones_cam1.json \
  --model yolov8n.pt \
  --conf 0.4 \
  --iou 0.45 \
  --output_dir ./runs \
  --use_ioa --ioa_thres 0.3  # optional
```

Key CLI options:
- `--recursive`: search for images in subfolders
- `--formats`: file extensions to include (default: jpg, jpeg, png)
- `--device`: force `cpu` or `cuda`

## Outputs
- **Visualizations**: saved to `runs/vis/` with polygons and bounding boxes (green = normal, red = violation, label shows class | score | zone names when violating).
- **CSV**: `runs/results.csv` with columns `image_name, cls, score, x1, y1, x2, y2, is_violation, zone_names`.
