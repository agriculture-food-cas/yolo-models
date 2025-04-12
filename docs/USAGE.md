# YOLO Object Detection Script

This script loads a YOLO model and uses it to detect objects in images.

## Requirements

Install the required packages:

```bash
pip install --no-index --find-links=wheels -r requirements.txt
```

## Usage

```bash
python model_evaluation.py --model [MODEL_PATH] --img [IMAGE_PATH] [OPTIONS]
```

### Arguments

- `--model`: Path to YOLO model file (default: `models/yolov8n.pt`)
- `--img`: Path to input image (required)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: NMS IoU threshold (default: 0.45)
- `--device`: Device to run inference on (`cpu` or `cuda`, default: `cpu`)
- `--output`: Output directory for results (default: `results`)
- `--view`: Display results (optional flag)

### Examples

Using YOLOv8 nano model:
```bash
python model_evaluation.py --model models/yolov8n.pt --img sample.jpg --conf 0.3
```

Using YOLOv5 model with visualization:
```bash
python model_evaluation.py --model models/yolov5s.pt --img sample.jpg --view
```

Using YOLO11 model with CUDA acceleration:
```bash
python model_evaluation.py --model models/yolo11n.pt --img sample.jpg --device cuda
```

Using YOLOv4 weights file:
```bash
python model_evaluation.py --model models/yolov4.weights --img sample.jpg
```

## Supported Models

The script supports different YOLO model formats:
- PyTorch models (.pt): YOLOv3, YOLOv5, YOLOv8, YOLO11
- Darknet models (.weights): YOLOv1, YOLOv4

## Output

Detection results are saved to the specified output directory (default: `results`). Each result image contains bounding boxes around detected objects with class labels and confidence scores.