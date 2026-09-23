import base64
import io
import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

# Keep third-party runtime settings inside this project so zipped/moved copies run cleanly.
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = PROJECT_ROOT / ".yolo_config"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
YOLO_CONFIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.inference import YOLOv11Inference
from src.utils import get_unique_classes_counts, load_metadata, save_metadata


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "coco-val-2017-500"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "coco-val-2017-500" / "metadata.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolo11m.pt"


st.set_page_config(page_title="YOLO COCO Visual Search", layout="wide")


def img_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def init_session_state() -> None:
    session_defaults: dict[str, Any] = {
        "metadata": None,
        "unique_classes": [],
        "count_options": {},
        "search_results": [],
        "search_ran": False,
        "search_params": {
            "search_mode": "Any selected class",
            "selected_classes": [],
            "thresholds": {},
        },
        "show_boxes": True,
        "grid_columns": 3,
        "highlight_matches": True,
        "active_device": None,
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_inferencer(model_path: str) -> YOLOv11Inference:
    return YOLOv11Inference(model_path)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #253244;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }

        .result-card {
            border: 1px solid #263241;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1rem;
            background: #111827;
        }

        .result-card img {
            display: block;
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: cover;
        }

        .result-meta {
            padding: 0.65rem 0.75rem;
            background: #0b111b;
            color: #f9fafb;
            font-size: 0.85rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def class_count_matches(item: dict[str, Any], selected_class: str, threshold: str) -> bool:
    class_count = sum(1 for detection in item["detections"] if detection["class"] == selected_class)
    if threshold == "None":
        return class_count >= 1
    return 1 <= class_count <= int(threshold)


def search_metadata(
    metadata: list[dict[str, Any]],
    selected_classes: list[str],
    thresholds: dict[str, str],
    search_mode: str,
) -> list[dict[str, Any]]:
    results = []
    require_all = search_mode == "All selected classes"

    for item in metadata:
        class_matches = [
            class_count_matches(item, selected_class, thresholds.get(selected_class, "None"))
            for selected_class in selected_classes
        ]

        if all(class_matches) if require_all else any(class_matches):
            results.append(item)

    return results


def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    selected_classes: list[str],
    highlight_matches: bool,
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for detection in detections:
        class_name = detection["class"]
        is_match = class_name in selected_classes

        if not is_match and highlight_matches:
            continue

        bbox = [int(value) for value in detection["bbox"]]
        color = "#30c938" if is_match else "#6b7280"
        thickness = 3 if is_match else 1
        label = f"{class_name} {detection['confidence']:.2f}"

        draw.rectangle(bbox, outline=color, width=thickness)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [bbox[0], bbox[1], bbox[0] + text_width + 8, bbox[1] + text_height + 4],
            fill=color,
        )
        draw.text((bbox[0] + 4, bbox[1] + 2), label, fill="white", font=font)

    return image


def render_result_card(result: dict[str, Any], selected_classes: list[str]) -> None:
    image = Image.open(result["image_path"]).convert("RGB")

    if st.session_state.show_boxes:
        image = draw_detections(
            image=image,
            detections=result["detections"],
            selected_classes=selected_classes,
            highlight_matches=st.session_state.highlight_matches,
        )

    meta_items = [
        f"{key}: {value}"
        for key, value in result["class_counts"].items()
        if key in selected_classes
    ]
    image_name = escape(Path(result["image_path"]).name)
    meta_text = escape(", ".join(meta_items) if meta_items else "No selected classes")

    st.markdown(
        f"""
        <div class="result-card">
            <img src="data:image/png;base64,{img_to_base64(image)}" alt="{image_name}">
            <div class="result-meta">
                <strong>{image_name}</strong><br>
                {meta_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


init_session_state()
render_styles()

st.title("YOLO COCO Visual Search")

option = st.radio(
    "Choose a workflow",
    ("Process new images", "Load existing metadata"),
    horizontal=True,
    label_visibility="collapsed",
)

if option == "Process new images":
    with st.expander("Process new images", expanded=True):
        image_col, model_col = st.columns(2)
        with image_col:
            image_dir = st.text_input(
                "Image directory path",
                value=str(DEFAULT_IMAGE_DIR) if DEFAULT_IMAGE_DIR.exists() else "",
                placeholder="path/to/images",
            )
        with model_col:
            model_path = st.text_input(
                "Model weights path",
                value=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else "yolo11m.pt",
            )

        if st.button("Start inference", type="primary"):
            if not image_dir:
                st.warning("Enter an image directory path.")
            else:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = get_inferencer(model_path)
                        metadata = inferencer.process_directory(image_dir)
                        metadata_path = save_metadata(metadata, image_dir)

                    st.session_state.metadata = metadata
                    st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                    st.session_state.search_results = []
                    st.session_state.search_ran = False
                    st.session_state.active_device = inferencer.device

                    st.success(f"Processed {len(metadata)} images on {inferencer.device.upper()}.")
                    st.code(str(metadata_path))
                except Exception as error:
                    st.error(f"Inference failed: {error}")
else:
    with st.expander("Load existing metadata", expanded=True):
        metadata_path = st.text_input(
            "Metadata file path",
            value=str(DEFAULT_METADATA_PATH) if DEFAULT_METADATA_PATH.exists() else "",
            placeholder="path/to/metadata.json",
        )

        if st.button("Load metadata", type="primary"):
            if not metadata_path:
                st.warning("Enter a metadata file path.")
            else:
                try:
                    with st.spinner("Loading metadata..."):
                        metadata = load_metadata(
                            metadata_path,
                            image_root=DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.exists() else None,
                        )

                    st.session_state.metadata = metadata
                    st.session_state.unique_classes, st.session_state.count_options = get_unique_classes_counts(metadata)
                    st.session_state.search_results = []
                    st.session_state.search_ran = False
                    st.session_state.active_device = "metadata"

                    st.success(f"Loaded metadata for {len(metadata)} images.")
                except Exception as error:
                    st.error(f"Metadata load failed: {error}")

if st.session_state.metadata:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Images indexed", len(st.session_state.metadata))
    metric_cols[1].metric("Detected classes", len(st.session_state.unique_classes))
    metric_cols[2].metric("Source", str(st.session_state.active_device or "ready").upper())

    st.header("Search")

    st.session_state.search_params["search_mode"] = st.radio(
        "Search mode",
        ("Any selected class", "All selected classes"),
        horizontal=True,
    )

    st.session_state.search_params["selected_classes"] = st.multiselect(
        "Classes",
        options=st.session_state.unique_classes,
    )

    selected_classes = st.session_state.search_params["selected_classes"]
    if selected_classes:
        threshold_cols = st.columns(min(len(selected_classes), 4))
        for index, class_name in enumerate(selected_classes):
            with threshold_cols[index % len(threshold_cols)]:
                st.session_state.search_params["thresholds"][class_name] = st.selectbox(
                    f"Max {class_name} count",
                    options=["None"] + st.session_state.count_options[class_name],
                )

    if st.button("Search images", type="primary"):
        if not selected_classes:
            st.warning("Select at least one class.")
        else:
            st.session_state.search_results = search_metadata(
                metadata=st.session_state.metadata,
                selected_classes=selected_classes,
                thresholds=st.session_state.search_params["thresholds"],
                search_mode=st.session_state.search_params["search_mode"],
            )
            st.session_state.search_ran = True

if st.session_state.search_ran:
    results = st.session_state.search_results
    selected_classes = st.session_state.search_params["selected_classes"]

    st.subheader(f"{len(results)} matching images")

    if results:
        with st.expander("Display options", expanded=True):
            box_col, grid_col, highlight_col = st.columns(3)
            with box_col:
                st.session_state.show_boxes = st.checkbox(
                    "Show bounding boxes",
                    value=st.session_state.show_boxes,
                )
            with grid_col:
                st.session_state.grid_columns = st.slider(
                    "Grid columns",
                    min_value=2,
                    max_value=6,
                    value=st.session_state.grid_columns,
                )
            with highlight_col:
                st.session_state.highlight_matches = st.checkbox(
                    "Highlight selected classes",
                    value=st.session_state.highlight_matches,
                )

        grid_cols = st.columns(st.session_state.grid_columns)
        for index, result in enumerate(results):
            with grid_cols[index % st.session_state.grid_columns]:
                try:
                    render_result_card(result, selected_classes)
                except Exception as error:
                    st.error(f"Could not display {result['image_path']}: {error}")

        with st.expander("Export"):
            st.download_button(
                label="Download results JSON",
                data=json.dumps(results, indent=2),
                file_name="search_results.json",
                mime="application/json",
            )
