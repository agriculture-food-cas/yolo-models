import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
import re

# Add YOLOv5, YOLOv3, YOLOv6 and Ultralytics to the path
YOLOV3_PATH = Path(__file__).parent / "yolov3"
YOLOV5_PATH = Path(__file__).parent / "yolov5"
YOLOV6_PATH = Path(__file__).parent / "YOLOv6"
ULTRALYTICS_PATH = Path(__file__).parent / "ultralytics"
if str(YOLOV3_PATH) not in sys.path:
    sys.path.append(str(YOLOV3_PATH))
if str(YOLOV5_PATH) not in sys.path:
    sys.path.append(str(YOLOV5_PATH))
if str(YOLOV6_PATH) not in sys.path:
    sys.path.append(str(YOLOV6_PATH))
if str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.append(str(ULTRALYTICS_PATH))

# Import YOLOv3 modules
try:
    from yolov3.models.common import DetectMultiBackend as YOLOv3DetectMultiBackend
    from yolov3.utils.general import (
        check_img_size as yolov3_check_img_size,
        non_max_suppression as yolov3_non_max_suppression,
        scale_boxes as yolov3_scale_boxes,
        xyxy2xywh as yolov3_xyxy2xywh,
    )
    from yolov3.utils.torch_utils import select_device as yolov3_select_device
    from yolov3.utils.augmentations import letterbox as yolov3_letterbox
    YOLOV3_AVAILABLE = True
except ImportError:
    YOLOV3_AVAILABLE = False
    LOGGER.warning("YOLOv3 not available. YOLOv3 models will not be supported.")

# Import YOLOv5 modules
try:
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import (
        check_img_size,
        non_max_suppression,
        scale_boxes,
        xyxy2xywh,
        cv2,
        LOGGER,
    )
    from yolov5.utils.torch_utils import select_device
    from yolov5.utils.augmentations import letterbox
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    LOGGER.warning("YOLOv5 not available. YOLOv5 models will not be supported.")

# Import YOLOv6 modules
try:
    from yolov6.models.yolo import build_model
    from yolov6.utils.config import Config
    from yolov6.utils.general import check_img_size as yolov6_check_img_size
    from yolov6.utils.nms import non_max_suppression as yolov6_non_max_suppression
    from yolov6.utils.general import xywh2xyxy as yolov6_xywh2xyxy
    YOLOV6_AVAILABLE = True
except ImportError:
    YOLOV6_AVAILABLE = False
    LOGGER.warning("YOLOv6 not available. YOLOv6 models will not be supported.")

# Import Ultralytics modules
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    LOGGER.warning("Ultralytics not available. Ultralytics models will not be supported.")


class YOLODetector:
    """
    A class for YOLO object detection that allows users to specify input and model.
    Supports YOLOv5, YOLOv3, YOLOv6, YOLOv8, and YOLOv12 models.
    """
    
    def __init__(
        self,
        model_path="yolov5s.pt",
        device="",
        img_size=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        classes=None,
        agnostic_nms=False,
        augment=False,
        half=False,
    ):
        """
        Initialize the YOLO detector with the specified model and parameters.
        
        Args:
            model_path (str): Path to the YOLO model weights file
            device (str): Device to run inference on (cuda device, i.e. 0 or 0,1,2,3 or cpu)
            img_size (tuple): Inference size (height, width)
            conf_thres (float): Confidence threshold
            iou_thres (float): NMS IOU threshold
            max_det (int): Maximum detections per image
            classes (list): Filter by class: [0, 15, 16] for COCO persons, cats and dogs
            agnostic_nms (bool): Class-agnostic NMS
            augment (bool): Augmented inference
            half (bool): Use FP16 half-precision inference
        """
        self.model_path = model_path  # Store the original model path
        self.device = select_device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.half = half
        self.img_size = img_size
        
        # Check if the model file exists
        if not os.path.exists(model_path) and not model_path.startswith(('http://', 'https://')):
            LOGGER.warning(f"Model file not found: {model_path}")
            LOGGER.info("Attempting to download the model from online...")
        
        # Determine model type and load appropriate model
        if model_path.endswith(('.pt', '.pth')):
            model_name = Path(model_path).stem.lower()
            
            # Check if it's an Ultralytics model (YOLOv8,v9,v10,11,12 or YOLOv5nu)
            if ('yolo12' in model_name or 
                'yolo11' in model_name or 
                'yolov10' in model_name or
                'yolov9' in model_name or
                'yolov8' in model_name or 
                'yolov7' in model_name or
                'yolov5nu' in model_name) and ULTRALYTICS_AVAILABLE:
                LOGGER.info(f"Loading Ultralytics model: {model_path}")
                try:
                    self.model_type = 'ultralytics'
                    self.model = YOLO(model_path)
                    self.names = self.model.names
                    LOGGER.info(f"Successfully loaded Ultralytics model")
                except Exception as e:
                    LOGGER.warning(f"Failed to load as Ultralytics model: {str(e)}")
                    # Try to download the model from online
                    downloaded_path = self._download_model_from_online(model_name)
                    if downloaded_path:
                        LOGGER.info(f"Successfully downloaded model: {downloaded_path}")
                        try:
                            self.model = YOLO(downloaded_path)
                            self.names = self.model.names
                            LOGGER.info(f"Successfully loaded downloaded model")
                        except Exception as load_error:
                            LOGGER.error(f"Failed to load downloaded model: {str(load_error)}")
                            self._handle_model_load_error(f"Failed to load model: {str(e)}")
                    else:
                        self._handle_model_load_error(f"Failed to load model: {str(e)}")
            # Check if it's a YOLOv6 model
            elif 'yolov6' in model_name and YOLOV6_AVAILABLE:
                LOGGER.info(f"Loading YOLOv6 model: {model_path}")
                try:
                    self.model_type = 'yolov6'
                    self._load_yolov6_model(model_path)
                    LOGGER.info(f"Successfully loaded YOLOv6 model")
                except Exception as e:
                    LOGGER.warning(f"Failed to load as YOLOv6 model: {str(e)}")
                    if ULTRALYTICS_AVAILABLE:
                        LOGGER.info("Falling back to Ultralytics")
                        try:
                            self._load_ultralytics_model(model_path)
                        except Exception as e2:
                            LOGGER.warning(f"Failed to load with Ultralytics: {str(e2)}")
                            # Try to download the model from online
                            downloaded_path = self._download_model_from_online(model_name)
                            if downloaded_path:
                                LOGGER.info(f"Successfully downloaded YOLOv6 model: {downloaded_path}")
                                try:
                                    self._load_yolov6_model(downloaded_path)
                                    LOGGER.info(f"Successfully loaded downloaded YOLOv6 model")
                                except Exception as load_error:
                                    LOGGER.error(f"Failed to load downloaded YOLOv6 model: {str(load_error)}")
                                    self._handle_model_load_error(f"Failed to load model: {str(e)}")
                            else:
                                self._handle_model_load_error(f"Failed to load model: {str(e)}")
            # Check if it's a YOLOv5 model
            elif 'yolov5' in model_name:
                LOGGER.info(f"Loading YOLOv5 model: {model_path}")
                try:
                    self._load_yolov5_model(model_path)
                    LOGGER.info(f"Successfully loaded YOLOv5 model")
                except Exception as e:
                    LOGGER.warning(f"Failed to load as YOLOv5 model: {str(e)}")
                    if ULTRALYTICS_AVAILABLE:
                        LOGGER.info("Falling back to Ultralytics")
                        try:
                            self._load_ultralytics_model(model_path)
                        except Exception as e2:
                            LOGGER.warning(f"Failed to load with Ultralytics: {str(e2)}")
                        # Try to download the model from online
                        downloaded_path = self._download_model_from_online(model_name)
                        if downloaded_path:
                            LOGGER.info(f"Successfully downloaded YOLOv5 model: {downloaded_path}")
                            try:
                                self._load_yolov5_model(downloaded_path)
                                LOGGER.info(f"Successfully loaded downloaded YOLOv5 model")
                            except Exception as load_error:
                                LOGGER.error(f"Failed to load downloaded YOLOv5 model: {str(load_error)}")
                                self._handle_model_load_error(f"Failed to load model: {str(e)}")
                        else:
                            self._handle_model_load_error(f"Failed to load model: {str(e)}")
            # Check if it's a YOLOv3 model
            elif 'yolov3' in model_name and YOLOV3_AVAILABLE:
                LOGGER.info(f"Loading YOLOv3 model: {model_path}")
                try:
                    self.model_type = 'yolov3'
                    self._load_yolov3_model(model_path)
                    LOGGER.info(f"Successfully loaded YOLOv3 model")
                except Exception as e:
                    LOGGER.warning(f"Failed to load as YOLOv3 model: {str(e)}")
                    if ULTRALYTICS_AVAILABLE:
                        LOGGER.info("Falling back to Ultralytics")
                        try:
                            self._load_ultralytics_model(model_path)
                        except Exception as e2:
                            LOGGER.warning(f"Failed to load with Ultralytics: {str(e2)}")
                            # Try to download the model from online
                            downloaded_path = self._download_model_from_online(model_name)
                            if downloaded_path:
                                LOGGER.info(f"Successfully downloaded YOLOv3 model: {downloaded_path}")
                                self._load_yolov3_model(downloaded_path)
                            else:
                                raise ValueError(f"Failed to load model: {str(e)}")
                    else:
                        # Try to download the model from online
                        downloaded_path = self._download_model_from_online(model_name)
                        if downloaded_path:
                            LOGGER.info(f"Successfully downloaded YOLOv3 model: {downloaded_path}")
                            self._load_yolov3_model(downloaded_path)
                        else:
                            raise ValueError(f"Failed to load model: {str(e)}")
            # Default to Ultralytics for all other models
            elif ULTRALYTICS_AVAILABLE:
                LOGGER.info(f"Loading model with Ultralytics: {model_path}")
                try:
                    self.model_type = 'ultralytics'
                    self.model = YOLO(model_path)
                    self.names = self.model.names
                    LOGGER.info(f"Successfully loaded model with Ultralytics")
                except Exception as e:
                    LOGGER.warning(f"Failed to load with Ultralytics: {str(e)}")
                    # Try to download the model from online
                    downloaded_path = self._download_model_from_online(model_name)
                    if downloaded_path:
                        LOGGER.info(f"Successfully downloaded model: {downloaded_path}")
                        self.model = YOLO(downloaded_path)
                        self.names = self.model.names
                    else:
                        raise ValueError(f"Failed to load model: {str(e)}")
            else:
                # Fallback to YOLOv5 if Ultralytics is not available
                LOGGER.warning(f"Failed to load with Ultralytics: {str(e)}")
                raise ValueError(f"Failed to load model: {str(e)}")
        else:
            raise ValueError("Unsupported model format. Use .pt or .pth files.")
        
        LOGGER.info(f"Loaded {len(self.names)} classes: {self.names}")
        
    def _download_model_from_online(self, model_name):
        """
        Download a YOLO model from online using Ultralytics.
        
        Args:
            model_name (str): Name of the model to download
            
        Returns:
            str: Path to the downloaded model file, or None if download failed
        """
        if not ULTRALYTICS_AVAILABLE:
            LOGGER.warning("Ultralytics not available. Cannot download model from online.")
            return None
            
        try:
            LOGGER.info(f"Attempting to download model: {model_name}")
            
            # Extract the model type and size from the model name
            # For example: 'yolov5s' -> type='yolov5', size='s'
            match = re.match(r'(yolo[v\d]+)([a-z]*)', model_name)
            if not match:
                LOGGER.warning(f"Could not parse model name: {model_name}")
                return None
                
            model_type, size = match.groups()
            
            # Map model types to their online identifiers
            # Note: Some model types might not be available in the Ultralytics library
            model_map = {
                'yolov3': 'yolov3',
                'yolov5': 'yolov5',
                'yolov6': 'yolov6',
                'yolov7': 'yolov7',
                'yolov8': 'yolov8',
                'yolov9': 'yolov9',
                'yolov10': 'yolov10',  # Make sure this matches the model name exactly
                'yolo10': 'yolov10',   # Alternative format
                'yolov11': 'yolo11',
                'yolo11': 'yolo11',
                'yolo12': 'yolov12'
            }
            
            # Check if the model type is in our mapping
            if model_type not in model_map:
                LOGGER.warning(f"Unknown model type: {model_type}")
                # Try to use the model name directly as a fallback
                model_id = model_name
                LOGGER.info(f"Using model name directly: {model_id}")
            else:
                # Construct the model identifier for download
                model_id = f"{model_map[model_type]}{size}"
            
            # Determine the save path based on the original model path
            # If the original path is a directory, save in that directory
            # Otherwise, save in the same directory as the original file
            original_path = Path(self.model_path)
            if original_path.is_dir():
                save_dir = original_path
            else:
                save_dir = original_path.parent
            
            # Create the directory if it doesn't exist
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the original filename if it's a file, otherwise use the model_id
            if original_path.is_file():
                save_path = save_dir / original_path.name
            else:
                save_path = save_dir / f"{model_id}.pt"
            
            # Download the model using Ultralytics
            LOGGER.info(f"Downloading model: {model_id} to {save_path}")
            
            try:
                # Try to download the model directly
                model = YOLO(model_id)
                
                # Try to save the model using different methods
                try:
                    # Method 1: Use the export method
                    model.export(format="torchscript", save=True, file=str(save_path))
                    LOGGER.info(f"Model exported to: {save_path}")
                except Exception as export_error:
                    LOGGER.warning(f"Failed to export model: {str(export_error)}")
                    try:
                        # Method 2: Try to save the model directly
                        model.save(str(save_path))
                        LOGGER.info(f"Model saved to: {save_path}")
                    except Exception as save_error:
                        LOGGER.warning(f"Failed to save model: {str(save_error)}")
                        try:
                            # Method 3: Try to save the state dict
                            torch.save(model.model.state_dict(), save_path)
                            LOGGER.info(f"Model state dict saved to: {save_path}")
                        except Exception as final_error:
                            LOGGER.warning(f"All attempts to save model failed: {str(final_error)}")
                            return None
                
                return str(save_path)
                
            except Exception as download_error:
                LOGGER.warning(f"Failed to download model {model_id}: {str(download_error)}")
                
                # Try alternative model names if the first attempt fails
                if model_type == 'yolov10' or model_type == 'yolo10':
                    # Try with 'yolov8' as a fallback for YOLOv10
                    alternative_id = f"yolov8{size}"
                    LOGGER.info(f"Trying alternative model: {alternative_id}")
                    try:
                        model = YOLO(alternative_id)
                        model.save(str(save_path))
                        LOGGER.info(f"Alternative model saved to: {save_path}")
                        return str(save_path)
                    except Exception as alt_error:
                        LOGGER.warning(f"Failed to download alternative model: {str(alt_error)}")
                
                return None
            
        except Exception as e:
            LOGGER.warning(f"Failed to download model: {str(e)}")
            return None
        
    def _load_yolov5_model(self, model_path):
        """Helper method to load YOLOv5 model"""
        LOGGER.info(f"Loading YOLOv5 model: {model_path}")
        self.model_type = 'yolov5'
        self.model = DetectMultiBackend(
            model_path, 
            device=self.device, 
            dnn=False, 
            data=None, 
            fp16=self.half
        )
        self.stride = self.model.stride
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.img_size))
        self.names = self.model.names
    
    def _load_yolov3_model(self, model_path):
        """Helper method to load YOLOv3 model"""
        LOGGER.info(f"Loading YOLOv3 model: {model_path}")
        self.model_type = 'yolov3'
        self.device = yolov3_select_device(device=self.device)
        self.model = YOLOv3DetectMultiBackend(
            model_path, 
            device=self.device, 
            dnn=False, 
            data=None, 
            fp16=self.half
        )
        self.stride = self.model.stride
        self.img_size = yolov3_check_img_size(self.img_size, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.img_size))
        self.names = self.model.names
    
    def _load_yolov6_model(self, model_path):
        """Helper method to load YOLOv6 model"""
        LOGGER.info(f"Loading YOLOv6 model: {model_path}")
        self.model_type = 'yolov6'
        
        # Default COCO class names - define this at the top for reuse
        default_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                         'hair drier', 'toothbrush']
        
        # Check if this is a segmentation model
        model_name = Path(model_path).stem.lower()
        is_seg_model = 'seg' in model_name
        
        if is_seg_model:
            LOGGER.warning(f"YOLOv6 segmentation model detected: {model_name}")
            LOGGER.warning("YOLOv6 segmentation models are not fully supported in this implementation.")
            LOGGER.warning("Falling back to Ultralytics for segmentation models.")
            if ULTRALYTICS_AVAILABLE:
                self._load_ultralytics_model(model_path)
                return
            else:
                raise ValueError("YOLOv6 segmentation models require Ultralytics, which is not available.")
        
        # Load model weights
        try:
            ckpt = torch.load(model_path, map_location=self.device)
            LOGGER.info(f"Loaded checkpoint type: {type(ckpt)}")
            
            # Check if the loaded model is already a Model object
            if hasattr(ckpt, 'forward') and hasattr(ckpt, 'detect'):
                LOGGER.info("Loaded model is already a YOLOv6 Model object")
                self.model = ckpt
                self.model.eval()
                
                # Set stride
                self.stride = max(int(self.model.detect.stride.max()), 32)
                
                # Set image size
                self.img_size = yolov6_check_img_size(self.img_size, s=self.stride)
                
                # Get class names - handle missing names attribute
                if hasattr(self.model, 'names'):
                    self.names = self.model.names
                else:
                    LOGGER.warning("Model object has no 'names' attribute, using default COCO class names")
                    self.names = default_names
                    # Add names attribute to the model for future use
                    self.model.names = default_names
                return
        except Exception as e:
            LOGGER.warning(f"Failed to load model file: {str(e)}")
            if ULTRALYTICS_AVAILABLE:
                LOGGER.info("Falling back to Ultralytics")
                try:
                    self._load_ultralytics_model(model_path)
                    return
                except Exception as ultralytics_error:
                    LOGGER.warning(f"Failed to load with Ultralytics: {str(ultralytics_error)}")
            else:
                LOGGER.warning("Ultralytics not available, continuing with YOLOv6 model building")
        
        # Get model configuration
        # For YOLOv6, we need to determine the config file based on the model name
        model_name = Path(model_path).stem.lower()
        
        # Map model names to config files
        config_map = {
            'yolov6n': 'yolov6n.py',
            'yolov6s': 'yolov6s.py',
            'yolov6m': 'yolov6m.py',
            'yolov6l': 'yolov6l.py',
            'yolov6x': 'yolov6x.py',
        }
        
        # Determine the config file
        config_name = None
        for key in config_map:
            if key in model_name:
                config_name = config_map[key]
                break
        
        if config_name is None:
            # Default to yolov6s if we can't determine the config
            config_name = 'yolov6s.py'
            LOGGER.warning(f"Could not determine config for model {model_name}, using default {config_name}")
        
        # Load the config file
        config_path = YOLOV6_PATH / 'configs' / config_name
        
        # Create a default config with all necessary attributes
        cfg = Config()
        cfg.model = {
            'backbone': {
                'type': 'EfficientRep', 
                'num_repeats': [1, 6, 12, 18, 6], 
                'out_channels': [64, 128, 256, 512, 1024],
                'use_se': False,
                'use_attn': False,
                'use_rep': True,
                'use_repblock': True,
                'repblock_mode': 'train'  # This is the key parameter that was missing
            },
            'neck': {
                'type': 'RepPAN', 
                'num_repeats': [12, 12, 12, 12], 
                'out_channels': [256, 128, 128, 256, 256, 512],
                'use_se': False,
                'use_attn': False,
                'use_rep': True,
                'use_repblock': True,
                'repblock_mode': 'train'  # This is the key parameter that was missing
            },
            'head': {
                'num_layers': 3, 
                'anchors': 1, 
                'strides': [8, 16, 32], 
                'reg_max': 16,
                'use_se': False,
                'use_attn': False,
                'use_rep': True,
                'use_repblock': True,
                'repblock_mode': 'train'  # This is the key parameter that was missing
            }
        }
        cfg.num_classes = 80  # COCO dataset has 80 classes
        cfg.training_mode = 'repvgg'  # Set to a valid string value for get_block function
        
        # Try to load the config file if it exists
        if os.path.exists(config_path):
            try:
                loaded_cfg = Config.fromfile(config_path)
                # Update our default config with values from the loaded config
                if hasattr(loaded_cfg, 'model'):
                    # Preserve our repblock_mode settings
                    if 'backbone' in loaded_cfg.model and 'repblock_mode' not in loaded_cfg.model['backbone']:
                        loaded_cfg.model['backbone']['repblock_mode'] = 'train'
                    if 'neck' in loaded_cfg.model and 'repblock_mode' not in loaded_cfg.model['neck']:
                        loaded_cfg.model['neck']['repblock_mode'] = 'train'
                    if 'head' in loaded_cfg.model and 'repblock_mode' not in loaded_cfg.model['head']:
                        loaded_cfg.model['head']['repblock_mode'] = 'train'
                    cfg.model = loaded_cfg.model
                if hasattr(loaded_cfg, 'num_classes'):
                    cfg.num_classes = loaded_cfg.num_classes
                if hasattr(loaded_cfg, 'training_mode'):
                    # Make sure training_mode is a valid string
                    if loaded_cfg.training_mode is False:
                        cfg.training_mode = 'repvgg'
                    else:
                        cfg.training_mode = loaded_cfg.training_mode
                LOGGER.info(f"Loaded config from {config_path}")
            except Exception as e:
                LOGGER.warning(f"Failed to load config file: {str(e)}, using default config")
        
        try:
            # Build the model
            self.model = build_model(cfg, cfg.num_classes, self.device)
            
            # Load the weights if available
            try:
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                    LOGGER.info("Successfully loaded model weights from state dict")
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                    LOGGER.info("Successfully loaded model weights from dict")
                else:
                    LOGGER.warning("Could not load weights from checkpoint, using initialized weights")
            except Exception as weight_error:
                LOGGER.warning(f"Failed to load weights: {str(weight_error)}, using initialized weights")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Set stride
            self.stride = max(int(self.model.detect.stride.max()), 32)
            
            # Set image size
            self.img_size = yolov6_check_img_size(self.img_size, s=self.stride)
            
            # Set class names
            self.names = default_names
            # Add names attribute to the model for future use
            self.model.names = default_names
            
        except Exception as e:
            LOGGER.warning(f"Failed to load YOLOv6 model: {str(e)}")
            if ULTRALYTICS_AVAILABLE:
                LOGGER.info("Falling back to Ultralytics")
                try:
                    self._load_ultralytics_model(model_path)
                except Exception as ultralytics_error:
                    LOGGER.error(f"Failed to load with Ultralytics: {str(ultralytics_error)}")
                    raise ValueError(f"Failed to load model: {str(e)}")
            else:
                raise ValueError(f"Failed to load YOLOv6 model: {str(e)}")
    
    def _load_ultralytics_model(self, model_path):
        """Helper method to load model with Ultralytics"""
        LOGGER.info(f"Loading model with Ultralytics: {model_path}")
        self.model_type = 'ultralytics'
        self.model = YOLO(model_path)
        self.names = self.model.names
        LOGGER.info(f"Successfully loaded model with Ultralytics")
    
    def preprocess_image(self, img):
        """
        Preprocess an image for inference.
        Only used for YOLOv5 and YOLOv3 models as Ultralytics handles preprocessing internally.
        """
        if self.model_type in ['ultralytics', 'yolov12', 'yolov11']:
            return img
            
        # YOLOv5/YOLOv3/YOLOv6 preprocessing
        if self.model_type == 'yolov3':
            im = yolov3_letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
        elif self.model_type == 'yolov6':
            # YOLOv6 uses the same letterbox function as YOLOv5
            im = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
        else:
            im = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]
            
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        return im
    
    def detect(self, source):
        """
        Perform detection on the specified source.
        
        Args:
            source (str): Path to image, video, directory, URL, glob, or webcam
            
        Returns:
            list: List of detection results
        """
        # Load source
        if isinstance(source, str):
            if os.path.isdir(source):
                files = sorted([os.path.join(source, f) for f in os.listdir(source) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            elif os.path.isfile(source):
                files = [source]
            else:
                raise FileNotFoundError(f"Source {source} does not exist")
        else:
            files = [source]
            
        results = []
        
        for file in files:
            # Read image
            img = cv2.imread(file)
            if img is None:
                LOGGER.warning(f"Failed to load image: {file}")
                continue
                
            if self.model_type in ['ultralytics']:
                # Ultralytics/YOLOv12/YOLOv11 inference
                predictions = self.model.predict(
                    source=img,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    max_det=self.max_det,
                    classes=self.classes,
                    agnostic_nms=self.agnostic_nms,
                    verbose=False
                )
                
                # Process Ultralytics/YOLOv12/YOLOv11/Yolov10/Yolov9/Yolov8/Yolov7/Yolov5 results
                for pred in predictions:
                    boxes = pred.boxes
                    for box in boxes:
                        class_idx = int(box.cls)
                        # Check if class_idx exists in names
                        if class_idx in self.names:
                            class_name = self.names[class_idx]
                        else:
                            class_name = f"class_{class_idx}"
                            LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                            
                        results.append({
                            'file': file,
                            'class': class_idx,
                            'class_name': class_name,
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        })
            elif self.model_type == 'yolov6':
                # YOLOv6 inference
                im = self.preprocess_image(img)
                pred = self.model(im)
                
                # Debug the model output and shapes
                LOGGER.info(f"YOLOv6 model output type: {type(pred)}")
                LOGGER.info(f"Input image shape: {img.shape}")
                LOGGER.info(f"Preprocessed image shape: {im.shape}")
                
                # Get the letterbox ratio and padding for coordinate scaling
                ratio, pad = letterbox(img, self.img_size, stride=self.stride, auto=True)[1:]
                LOGGER.info(f"Letterbox ratio: {ratio}, pad: {pad}")
                
                # Process YOLOv6 output
                results = []
                if isinstance(pred, list) and len(pred) > 0:
                    # The first element contains the main detection output
                    main_pred = pred[0]
                    if isinstance(main_pred, torch.Tensor):
                        LOGGER.info(f"Main prediction shape: {main_pred.shape}")
                        LOGGER.info(f"Main prediction content sample: {main_pred[0, 0, :10]}")
                        
                        # Apply NMS to the main detection output
                        main_pred = yolov6_non_max_suppression(
                            main_pred,
                            self.conf_thres,
                            self.iou_thres,
                            self.classes,
                            self.agnostic_nms,
                            max_det=self.max_det
                        )
                        
                        LOGGER.info(f"NMS output type: {type(main_pred)}")
                        if isinstance(main_pred, list):
                            LOGGER.info(f"NMS output length: {len(main_pred)}")
                        
                        # Process detections
                        for i, det in enumerate(main_pred):
                            if len(det):
                                LOGGER.info(f"Detection {i} shape: {det.shape}")
                                LOGGER.info(f"Detection {i} content: {det[0]}")
                                
                                # Rescale boxes from img_size to original image size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape, (ratio, pad)).round()
                                LOGGER.info(f"Scaled boxes: {det[:, :4]}")
                                
                                for *xyxy, conf, cls in reversed(det):
                                    class_idx = int(cls)
                                    # Check if class_idx exists in names
                                    if class_idx in self.names:
                                        class_name = self.names[class_idx]
                                    else:
                                        class_name = f"class_{class_idx}"
                                        LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                                    
                                    # Ensure bounding box coordinates are valid
                                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                                    LOGGER.info(f"Raw coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                    
                                    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                                        LOGGER.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                                        # Clip coordinates to image boundaries
                                        x1 = max(0, min(x1, img.shape[1]))
                                        y1 = max(0, min(y1, img.shape[0]))
                                        x2 = max(0, min(x2, img.shape[1]))
                                        y2 = max(0, min(y2, img.shape[0]))
                                        LOGGER.info(f"Clipped coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                    
                                    # Check for zero-width or zero-height boxes
                                    if x2 <= x1 or y2 <= y1:
                                        LOGGER.warning(f"Zero-width or zero-height bounding box: [{x1}, {y1}, {x2}, {y2}]")
                                        # Skip this detection
                                        continue
                                    
                                    results.append({
                                        'file': file,
                                        'class': class_idx,
                                        'class_name': class_name,
                                        'confidence': float(conf),
                                        'bbox': [x1, y1, x2, y2]
                                    })
                else:
                    LOGGER.warning("YOLOv6 model output is not in the expected format")
                
                LOGGER.info(f"Total detections found: {len(results)}")
            else:
                # YOLOv5/YOLOv3 inference
                im = self.preprocess_image(img)
                pred = self.model(im, augment=self.augment)
                
                # Apply NMS
                if self.model_type == 'yolov3':
                    pred = yolov3_non_max_suppression(
                        pred, 
                        self.conf_thres, 
                        self.iou_thres, 
                        self.classes, 
                        self.agnostic_nms, 
                        max_det=self.max_det
                    )
                else:
                    pred = non_max_suppression(
                        pred, 
                        self.conf_thres, 
                        self.iou_thres, 
                        self.classes, 
                        self.agnostic_nms, 
                        max_det=self.max_det
                    )
                
                # Process YOLOv5/YOLOv3 results
                for i, det in enumerate(pred):
                    if len(det):
                        # Rescale boxes from img_size to original image size
                        if self.model_type == 'yolov3':
                            det[:, :4] = yolov3_scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                        else:
                            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                            
                        for *xyxy, conf, cls in reversed(det):
                            class_idx = int(cls)
                            # Check if class_idx exists in names
                            if class_idx in self.names:
                                class_name = self.names[class_idx]
                            else:
                                class_name = f"class_{class_idx}"
                                LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                                
                            results.append({
                                'file': file,
                                'class': class_idx,
                                'class_name': class_name,
                                'confidence': float(conf),
                                'bbox': [float(x) for x in xyxy]
                            })
        
        return results
    
    def detect_image(self, img):
        """
        Perform detection on a single image.
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            list: List of detection results
        """
        # Preprocess image
        im = self.preprocess_image(img)
        
        # Inference
        if self.model_type in ['ultralytics', 'yolov12', 'yolov11']:
            # Ultralytics/YOLOv12/YOLOv11 inference
            predictions = self.model.predict(
                source=img,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                classes=self.classes,
                agnostic_nms=self.agnostic_nms,
                verbose=False
            )
            
            # Process Ultralytics/YOLOv12/YOLOv11 results
            results = []
            for pred in predictions:
                boxes = pred.boxes
                for box in boxes:
                    class_idx = int(box.cls)
                    # Check if class_idx exists in names
                    if class_idx in self.names:
                        class_name = self.names[class_idx]
                    else:
                        class_name = f"class_{class_idx}"
                        LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                        
                    results.append({
                        'class': class_idx,
                        'class_name': class_name,
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    })
            return results
        elif self.model_type == 'yolov6':
            # YOLOv6 inference
            pred = self.model(im)
            
            # Debug the model output
            LOGGER.info(f"YOLOv6 model output type: {type(pred)}")
            if isinstance(pred, list):
                LOGGER.info(f"YOLOv6 model output length: {len(pred)}")
                for i, p in enumerate(pred):
                    LOGGER.info(f"Output {i} type: {type(p)}")
                    if isinstance(p, list):
                        LOGGER.info(f"Output {i} length: {len(p)}")
                        for j, sub_p in enumerate(p):
                            LOGGER.info(f"Output {i}.{j} type: {type(sub_p)}")
                            if isinstance(sub_p, torch.Tensor):
                                LOGGER.info(f"Output {i}.{j} shape: {sub_p.shape}")
                                LOGGER.info(f"Output {i}.{j} content sample: {sub_p[0, 0, :10]}")
            
            # Handle different output formats from YOLOv6
            if isinstance(pred, list):
                # YOLOv6 returns a list of tensors, one for each detection head
                LOGGER.info("YOLOv6 model returned a list of tensors")
                
                # Process each detection head output
                results = []
                for i, p in enumerate(pred):
                    # Handle nested list structure
                    if isinstance(p, list):
                        LOGGER.info(f"Processing nested list at index {i} with {len(p)} elements")
                        for j, sub_p in enumerate(p):
                            if isinstance(sub_p, torch.Tensor):
                                # Debug the tensor shape and content
                                LOGGER.info(f"Tensor shape: {sub_p.shape}")
                                LOGGER.info(f"Tensor content sample: {sub_p[0, 0, :10]}")
                                
                                # Apply NMS to each detection head output
                                sub_p = yolov6_non_max_suppression(
                                    sub_p, 
                                    self.conf_thres, 
                                    self.iou_thres, 
                                    self.classes, 
                                    self.agnostic_nms, 
                                    max_det=self.max_det
                                )
                                
                                # Debug the NMS output
                                LOGGER.info(f"NMS output type: {type(sub_p)}")
                                if isinstance(sub_p, list):
                                    LOGGER.info(f"NMS output length: {len(sub_p)}")
                                    for k, det in enumerate(sub_p):
                                        if len(det):
                                            LOGGER.info(f"Detection {k} shape: {det.shape}")
                                            LOGGER.info(f"Detection {k} content: {det[0]}")
                                
                                # Process detections from this head
                                for k, det in enumerate(sub_p):
                                    if len(det):
                                        # Get the letterbox ratio and padding
                                        ratio, pad = letterbox(img, self.img_size, stride=self.stride, auto=True)[1:]
                                        LOGGER.info(f"Letterbox ratio: {ratio}, pad: {pad}")
                                        
                                        # Rescale boxes from img_size to original image size
                                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape, (ratio, pad)).round()
                                        LOGGER.info(f"Scaled boxes: {det[:, :4]}")
                                        
                                        for *xyxy, conf, cls in reversed(det):
                                            class_idx = int(cls)
                                            # Check if class_idx exists in names
                                            if class_idx in self.names:
                                                class_name = self.names[class_idx]
                                            else:
                                                class_name = f"class_{class_idx}"
                                                LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                                                
                                            # Ensure bounding box coordinates are valid
                                            x1, y1, x2, y2 = [float(x) for x in xyxy]
                                            LOGGER.info(f"Raw coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                            
                                            if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                                                LOGGER.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                                                # Clip coordinates to image boundaries
                                                x1 = max(0, min(x1, img.shape[1]))
                                                y1 = max(0, min(y1, img.shape[0]))
                                                x2 = max(0, min(x2, img.shape[1]))
                                                y2 = max(0, min(y2, img.shape[0]))
                                                LOGGER.info(f"Clipped coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                                
                                            # Check for zero-width or zero-height boxes
                                            if x2 <= x1 or y2 <= y1:
                                                LOGGER.warning(f"Zero-width or zero-height bounding box: [{x1}, {y1}, {x2}, {y2}]")
                                                # Skip this detection
                                                continue
                                                    
                                            results.append({
                                                'file': file,
                                                'class': class_idx,
                                                'class_name': class_name,
                                                'confidence': float(conf),
                                                'bbox': [x1, y1, x2, y2]
                                            })
                    elif isinstance(p, torch.Tensor):
                        # Debug the tensor shape and content
                        LOGGER.info(f"Tensor shape: {p.shape}")
                        LOGGER.info(f"Tensor content sample: {p[0, 0, :10]}")
                        
                        # Apply NMS to each detection head output
                        p = yolov6_non_max_suppression(
                            p, 
                            self.conf_thres, 
                            self.iou_thres, 
                            self.classes, 
                            self.agnostic_nms, 
                            max_det=self.max_det
                        )
                        
                        # Debug the NMS output
                        LOGGER.info(f"NMS output type: {type(p)}")
                        if isinstance(p, list):
                            LOGGER.info(f"NMS output length: {len(p)}")
                            for j, det in enumerate(p):
                                if len(det):
                                    LOGGER.info(f"Detection {j} shape: {det.shape}")
                                    LOGGER.info(f"Detection {j} content: {det[0]}")
                        
                        # Process detections from this head
                        for j, det in enumerate(p):
                            if len(det):
                                # Rescale boxes from img_size to original image size
                                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                                
                                for *xyxy, conf, cls in reversed(det):
                                    class_idx = int(cls)
                                    # Check if class_idx exists in names
                                    if class_idx in self.names:
                                        class_name = self.names[class_idx]
                                    else:
                                        class_name = f"class_{class_idx}"
                                        LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                                        
                                    # Ensure bounding box coordinates are valid
                                    x1, y1, x2, y2 = [float(x) for x in xyxy]
                                    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                                        LOGGER.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                                        # Clip coordinates to image boundaries
                                        x1 = max(0, min(x1, img.shape[1]))
                                        y1 = max(0, min(y1, img.shape[0]))
                                        x2 = max(0, min(x2, img.shape[1]))
                                        y2 = max(0, min(y2, img.shape[0]))
                                        
                                    # Check for zero-width or zero-height boxes
                                    if x2 <= x1 or y2 <= y1:
                                        LOGGER.warning(f"Zero-width or zero-height bounding box: [{x1}, {y1}, {x2}, {y2}]")
                                        # Skip this detection
                                        continue
                                        
                                    results.append({
                                        'class': class_idx,
                                        'class_name': class_name,
                                        'confidence': float(conf),
                                        'bbox': [x1, y1, x2, y2]
                                    })
                return results
            else:
                # Handle tensor output (original format)
                # Debug the tensor shape and content
                LOGGER.info(f"Tensor shape: {pred.shape}")
                LOGGER.info(f"Tensor content sample: {pred[0, 0, :10]}")
                
                # Apply NMS
            pred = yolov6_non_max_suppression(
                pred, 
                self.conf_thres, 
                self.iou_thres, 
                self.classes, 
                self.agnostic_nms, 
                max_det=self.max_det
            )
            
            # Debug the NMS output
            LOGGER.info(f"NMS output type: {type(pred)}")
            if isinstance(pred, list):
                LOGGER.info(f"NMS output length: {len(pred)}")
                for i, det in enumerate(pred):
                    if len(det):
                        LOGGER.info(f"Detection {i} shape: {det.shape}")
                        LOGGER.info(f"Detection {i} content: {det[0]}")
                
            # Process YOLOv6 results
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                    
                    for *xyxy, conf, cls in reversed(det):
                        class_idx = int(cls)
                        # Check if class_idx exists in names
                        if class_idx in self.names:
                            class_name = self.names[class_idx]
                        else:
                            class_name = f"class_{class_idx}"
                            LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                            
                            # Ensure bounding box coordinates are valid
                            x1, y1, x2, y2 = [float(x) for x in xyxy]
                            if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                                LOGGER.warning(f"Invalid bounding box coordinates: [{x1}, {y1}, {x2}, {y2}]")
                                # Clip coordinates to image boundaries
                                x1 = max(0, min(x1, img.shape[1]))
                                y1 = max(0, min(y1, img.shape[0]))
                                x2 = max(0, min(x2, img.shape[1]))
                                y2 = max(0, min(y2, img.shape[0]))
                                
                            # Check for zero-width or zero-height boxes
                            if x2 <= x1 or y2 <= y1:
                                LOGGER.warning(f"Zero-width or zero-height bounding box: [{x1}, {y1}, {x2}, {y2}]")
                                # Skip this detection
                                continue
                            
                        results.append({
                            'class': class_idx,
                            'class_name': class_name,
                            'confidence': float(conf),
                                'bbox': [x1, y1, x2, y2]
                        })
            
            return results
        else:
            # YOLOv5/YOLOv3 inference
            pred = self.model(im, augment=self.augment)
            
            # NMS
            if self.model_type == 'yolov3':
                pred = yolov3_non_max_suppression(
                    pred, 
                    self.conf_thres, 
                    self.iou_thres, 
                    self.classes, 
                    self.agnostic_nms, 
                    max_det=self.max_det
                )
            else:
                pred = non_max_suppression(
                    pred, 
                    self.conf_thres, 
                    self.iou_thres, 
                    self.classes, 
                    self.agnostic_nms, 
                    max_det=self.max_det
                )
            
            # Process detections
            results = []
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to original image size
                    if self.model_type == 'yolov3':
                        det[:, :4] = yolov3_scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                    else:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
                    
                    # Convert results to list of dictionaries
                    for *xyxy, conf, cls in reversed(det):
                        class_idx = int(cls)
                        # Check if class_idx exists in names
                        if class_idx in self.names:
                            class_name = self.names[class_idx]
                        else:
                            class_name = f"class_{class_idx}"
                            LOGGER.warning(f"Class index {class_idx} not found in model names. Using default name.")
                            
                        results.append({
                            'class': class_idx,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x) for x in xyxy]  # [x1, y1, x2, y2]
                        })
            
            return results
    
    def draw_detections(self, img, detections):
        """
        Draw detection results on the image.
        
        Args:
            img (numpy.ndarray): Input image
            detections (list): List of detection results
            
        Returns:
            numpy.ndarray: Image with detections drawn
        """
        img_copy = img.copy()
        
        for det in detections:
            # Get bbox coordinates
            x1, y1, x2, y2 = det['bbox']
            
            # Get class name and confidence
            class_name = det['class_name']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (int(x1), int(y1) - label_height - 10), (int(x1) + label_width, int(y1)), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_copy

    def _handle_model_load_error(self, error_message):
        """
        Handle model loading errors with a graceful exit.
        
        Args:
            error_message (str): The error message to display
        """
        LOGGER.error(f"ERROR: {error_message}")
        LOGGER.error("Model loading failed. Please check the model path and try again.")
        LOGGER.error("If you're using a custom model, make sure it's compatible with the YOLO format.")
        LOGGER.error("For more information, see the documentation: https://docs.ultralytics.com/")
        
        # Exit the program with a non-zero status code
        sys.exit(1)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    
    # Model parameters
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='inference size (height, width)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    
    # Input/output parameters
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--save-dir', type=str, default='runs/detect', help='save results to project/name')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--view-img', action='store_true', help='show results')
    
    # Visualization parameters
    parser.add_argument('--line-thickness', type=int, default=3, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='hide confidences')
    
    return parser.parse_args()


def main(args):
    """
    Main function to run YOLO detection.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Initialize detector
    detector = YOLODetector(
        model_path=args.weights,
        device=args.device,
        img_size=tuple(args.img_size),
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        classes=args.classes,
        agnostic_nms=args.agnostic_nms,
        augment=args.augment,
        half=args.half,
    )
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect objects
    results = detector.detect(args.source)
    
    # Group results by file
    results_by_file = {}
    for result in results:
        file_path = result['file']
        if file_path not in results_by_file:
            results_by_file[file_path] = []
        results_by_file[file_path].append(result)
    
    # Process results for each file
    for file_path, file_results in results_by_file.items():
        # Print detection results
        for result in file_results:
            print(f"File: {result['file']}")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Bounding Box: {result['bbox']}")
            print("---")
        
        # Save results if needed
        if not args.nosave:
            # Read image
            img = cv2.imread(file_path)
            
            # Draw all detections for this image at once
            img_with_detections = detector.draw_detections(img, file_results)
            
            # Save image
            save_path = save_dir / Path(file_path).name
            cv2.imwrite(str(save_path), img_with_detections)
            
            # Save text results if needed
            if args.save_txt:
                txt_path = save_dir / Path(file_path).stem + '.txt'
                with open(txt_path, 'w') as f:  # Changed from 'a' to 'w' to avoid duplicates
                    for result in file_results:
                        # Format: class x_center y_center width height confidence
                        x1, y1, x2, y2 = result['bbox']
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        if args.save_conf:
                            f.write(f"{result['class']} {x_center} {y_center} {width} {height} {result['confidence']}\n")
                        else:
                            f.write(f"{result['class']} {x_center} {y_center} {width} {height}\n")
            
            # Show image if needed
            if args.view_img:
                cv2.imshow("Detections", img_with_detections)
                cv2.waitKey(0)
    
    print(f"Detection completed. Results saved to {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
