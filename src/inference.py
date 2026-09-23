import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_CONFIG_DIR = PROJECT_ROOT / ".yolo_config"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
YOLO_CONFIG_DIR.mkdir(exist_ok=True)

from ultralytics import YOLO
import torch
# from PIL import Image
# import numpy
from src.config import load_config


logger = logging.getLogger(__name__)


def _project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class YOLOv11Inference:
    def __init__(self, model_name, device="auto"):
        self.model_name = self._resolve_model_name(model_name)
        self.model = YOLO(self.model_name)
        self.device = self._select_device(device)
        self.model.to(self.device)

        # loading config from default.yaml
        config = load_config()
        self.conf_threshold = config["model"]["conf_threshold"]
        self.extensions = config["data"]["image_extension"]

    @staticmethod
    def _resolve_model_name(model_name):
        model_path = Path(model_name).expanduser()
        if model_path.is_absolute() and model_path.exists():
            return str(model_path)

        project_model_path = PROJECT_ROOT / model_path
        if project_model_path.exists():
            return str(project_model_path)

        return str(model_name)

    @staticmethod
    def _select_device(device):
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def process_image(self, image_path):
        image_path = _project_path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            device=self.device
        )

        # process results
        detection = []
        class_counts = {}

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                detection.append({
                    'class' : cls,
                    'confidence' : conf,
                    'bbox' : bbox,
                    'count' : 1
                })

                class_counts[cls] = class_counts.get(cls, 0) + 1

        for det in detection:
            det['count'] = class_counts[det['class']]

        return {
            'image_path' : str(image_path.resolve()),
            'detections' : detection,
            'total_objects' : len(detection),
            'unique_class' : list(class_counts.keys()), # [0, 1, 2]
            'class_counts' : class_counts # {0 : 3, 1 : 10, 2, : 1}
        }


    def process_directory(self, directory):
        metadata = []
        directory = _project_path(directory).resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Image directory not found: {directory}")

        extensions = {extension.lower() for extension in self.extensions}
        image_paths = sorted(
            image_path
            for image_path in directory.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in extensions
        )

        if not image_paths:
            raise FileNotFoundError(f"No supported images found in: {directory}")

        for img_path in image_paths:
            try:
                metadata.append(self.process_image(img_path))
            except Exception as e:
                logger.warning("Error processing %s: %s", img_path, e)
                continue
        # print(metadata)
        return metadata
