  #!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO model inference on images')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--img', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--view', action='store_true', help='Display results')
    return parser.parse_args()

def detect_objects(args):
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model {args.model} not found")
    
    # Check if image exists
    if not os.path.exists(args.img):
        raise FileNotFoundError(f"Image {args.img} not found")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine model type from extension
    model_ext = os.path.splitext(args.model)[1].lower()
    
    # Load model based on type
    if model_ext == '.pt':  # PyTorch models (YOLOv3, YOLOv5, YOLOv8, YOLO11)
        try:
            import torch
            
            # Try to load using Ultralytics YOLOv8/YOLO11 API first
            try:
                from ultralytics import YOLO
                model = YOLO(args.model)
                
                # Run inference
                start_time = time.time()
                results = model(args.img, conf=args.conf, iou=args.iou, device=args.device)
                inference_time = time.time() - start_time
                
                # Process results
                for i, result in enumerate(results):
                    im_array = result.plot()  # Plot results image
                    im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
                    
                    # Save image
                    img_name = os.path.basename(args.img)
                    save_path = os.path.join(args.output, f"result_{img_name}")
                    im.save(save_path)
                    
                    # Display results information
                    print(f"Detected {len(result.boxes)} objects")
                    print(f"Inference time: {inference_time:.2f} seconds")
                    
                    # Display image if requested
                    if args.view:
                        im.show()
                
                return
            
            # Fallback to YOLOv5 loading method
            except (ImportError, Exception) as e:
                print(f"Ultralytics YOLO loading error: {e}")
                print("Attempting to load with PyTorch...")
                
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.model)
                model.conf = args.conf
                model.iou = args.iou
                
                # Run inference
                start_time = time.time()
                results = model(args.img)
                inference_time = time.time() - start_time
                
                # Save results
                results.save(save_dir=args.output)
                
                # Display results
                print(f"Detected {len(results.xyxy[0])} objects")
                print(f"Inference time: {inference_time:.2f} seconds")
                
                # Display image if requested
                if args.view:
                    results.show()
                
                return
                
        except ImportError:
            print("PyTorch and/or Ultralytics not found. Please install with: pip install torch ultralytics")
            return
    
    elif model_ext == '.weights':  # Darknet models (YOLOv1, YOLOv2, YOLOv4)
        try:
            # Try to use OpenCV DNN for Darknet models
            # Determine if it's YOLOv4/v3 or older
            model_name = os.path.basename(args.model).lower()
            
            # Load images
            img = cv2.imread(args.img)
            height, width = img.shape[:2]
            
            # Define configuration and weights files based on model name
            config_file = None
            if 'yolov4' in model_name:
                config_file = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
                if 'tiny' in model_name:
                    config_file = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
            elif 'yolov3' in model_name:
                config_file = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
                if 'tiny' in model_name:
                    config_file = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
            elif 'yolov1' in model_name:
                config_file = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1.cfg'
                if 'tiny' in model_name:
                    config_file = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov1-tiny.cfg'
            
            if config_file is None:
                print(f"Could not determine configuration file for {args.model}")
                return
            
            # Load YOLO model with OpenCV
            net = cv2.dnn.readNetFromDarknet(config_file, args.model)
            
            # Set preferred backend
            if args.device == 'cuda':
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            # COCO class names
            classes = []
            with open('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Create blob and perform inference
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            
            start_time = time.time()
            outs = net.forward(output_layers)
            inference_time = time.time() - start_time
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > args.conf:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.iou)
            
            # Draw bounding boxes
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save result
            img_name = os.path.basename(args.img)
            save_path = os.path.join(args.output, f"result_{img_name}")
            cv2.imwrite(save_path, img)
            
            # Display results
            print(f"Detected {len(indexes)} objects")
            print(f"Inference time: {inference_time:.2f} seconds")
            
            # Display image if requested
            if args.view:
                cv2.imshow("Result", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return
            
        except Exception as e:
            print(f"OpenCV loading error: {e}")
            print("Failed to load Darknet model with OpenCV")
            return
    
    else:
        print(f"Unsupported model format: {model_ext}")
        return

def main():
    args = parse_args()
    detect_objects(args)

if __name__ == "__main__":
    main()