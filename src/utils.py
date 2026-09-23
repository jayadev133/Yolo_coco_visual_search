import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_processed_dir(raw_path):
    raw_path = _project_path(raw_path).resolve()
    # "c:/abc/def/image_01000" --> raw_path.name --> image_01000
    processed_path = PROJECT_ROOT / "data" / "processed" / raw_path.name
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path


def save_metadata(metadata, raw_path):
    processed_path = ensure_processed_dir(raw_path)

    output_path = processed_path / "metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output_path


def _metadata_candidates(metadata_path):
    metadata_path = _project_path(metadata_path)
    yield metadata_path

    if metadata_path.suffix.lower() != ".json":
        yield metadata_path / "metadata.json"
        yield PROJECT_ROOT / "data" / "processed" / metadata_path.name / "metadata.json"


def _resolve_metadata_path(metadata_path):
    for candidate in _metadata_candidates(metadata_path):
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Metadata not found at {metadata_path}")


def _image_candidates(image_path, metadata_path, image_root=None):
    image_path = Path(image_path)
    filename = image_path.name

    if image_path.exists():
        yield image_path

    if image_root:
        yield _project_path(image_root) / filename

    dataset_name = metadata_path.parent.name
    for parent in metadata_path.parents:
        yield parent / dataset_name / filename
        yield parent / "raw" / dataset_name / filename
        yield parent / "data" / "raw" / dataset_name / filename


def _resolve_image_path(image_path, metadata_path, image_root=None):
    for candidate in _image_candidates(image_path, metadata_path, image_root):
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return str(image_path)


def repair_metadata_image_paths(metadata, metadata_path, image_root=None):
    for item in metadata:
        if "image_path" in item:
            item["image_path"] = _resolve_image_path(
                item["image_path"],
                metadata_path,
                image_root=image_root
            )
    return metadata


def load_metadata(metadata_path, image_root=None):
    metadata_path = _resolve_metadata_path(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return repair_metadata_image_paths(metadata, metadata_path, image_root=image_root)
        

def get_unique_classes_counts(metadata):
    unique_classes = set()
    count_options = {}

    for item in metadata:
        for cls in item['detections']:
            unique_classes.add(cls['class'])
            if cls['class'] not in count_options:
                count_options[cls['class']]= set()
            count_options[cls['class']].add(cls['count'])

    unique_classes = sorted(unique_classes)
    for cls in count_options:
        count_options[cls] = sorted(count_options[cls])
    
    return unique_classes, count_options

