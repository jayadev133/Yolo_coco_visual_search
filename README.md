# YOLO COCO Visual Search

A Streamlit application for indexing image folders with YOLO detections and searching them by object class. The app can run fresh inference with Ultralytics YOLO or load previously generated metadata for fast interactive search.

## Features

- Process `.jpg`, `.jpeg`, and `.png` image folders with YOLO.
- Automatically uses CUDA when a compatible NVIDIA GPU and PyTorch build are available.
- Search with OR or AND matching across detected COCO classes.
- Add optional maximum-count filters per class.
- Review results in a responsive image grid with bounding boxes.
- Export search results as JSON.

## Project Structure

```text
.
|-- app.py
|-- configs/
|   `-- default.yaml
|-- src/
|   |-- config.py
|   |-- inference.py
|   `-- utils.py
|-- requirements.txt
`-- README.md
```

Large local artifacts such as model weights, COCO images, generated metadata, and course PDFs are intentionally excluded from Git. Keep those files locally or publish them separately through a release, cloud bucket, or data registry.

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU and CUDA-capable PyTorch build for GPU inference
- CPU inference also works, but processing large image folders will be slower

## Setup

Create and activate an environment:

```powershell
conda create -n yolo_image_search python=3.11 -y
conda activate yolo_image_search
```

For GPU usage, install the CUDA-enabled PyTorch build that matches your system before installing the app dependencies. On a CUDA 12.6 setup, for example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

For CPU-only usage:

```powershell
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Running the App

Start Streamlit from the project root:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints in the terminal, usually:

```text
http://localhost:8501
```

## Model Weights

The app accepts either a local model path or a standard Ultralytics model name.

Examples:

```text
yolo11m.pt
C:\path\to\weights\yolo11m.pt
```

If the model file is in the project root, the app will resolve it automatically. The repository ignores `*.pt` files so large model weights do not get committed by accident.

## Image Data

Use an image folder with supported image files:

```text
C:\path\to\coco-val-2017-500
```

After inference, metadata is saved under:

```text
data\processed\<image-folder-name>\metadata.json
```

You can later choose **Load existing metadata** in the app and point to that JSON file to skip inference.

## Configuration

Default inference settings live in:

```text
configs\default.yaml
```

Current options:

```yaml
model:
  yolo_model: "yolo11m.pt"
  conf_threshold: 0.3

data:
  image_extension: [".jpg", ".jpeg", ".png"]
```

## Screenshots

<img width="1302" height="207" alt="image" src="https://github.com/user-attachments/assets/79d35195-bf32-4dea-bb7e-5d16450e64e7" />

<img width="1290" height="120" alt="image" src="https://github.com/user-attachments/assets/2ac09174-f871-40f3-b71f-140e2398faef" />

<img width="1908" height="893" alt="image" src="https://github.com/user-attachments/assets/9142a37f-6b8d-4ea3-99c8-b9fddeedd597" />

<img width="1917" height="947" alt="image" src="https://github.com/user-attachments/assets/2b7bfd25-ccc3-4f25-b2ea-2d98449de890" />

<img width="1887" height="868" alt="image" src="https://github.com/user-attachments/assets/efe8f5cf-d769-4a5b-8e69-fa2c2c36add1" />



## Notes for Deployment

- Do not commit model weights, local datasets, generated metadata, or secrets.
- Install the correct PyTorch build for your deployment target before running the app.
- For hosted environments, mount model weights and datasets as external volumes or download them during startup.
- For repeatable demos, generate metadata once and load it directly from the app.
