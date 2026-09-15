import os

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from config import COCO_AIRPLANE_ID, COCO_BIRD_ID
from tracking import detection_iou, detections_match


# Returns: filtered predictions minus edge-touching detections.

def filter_border_detections(predictions, image_width, image_height, margin_ratio):
    """Rejects predictions that touch the image border entirely.

    Objects cut by real-image edges have less visible context and are more
    likely to be confused with unrelated shapes such as lamp posts.
    """
    margin_x = image_width * margin_ratio
    margin_y = image_height * margin_ratio

    kept = []
    for p in predictions:
        min_x, min_y, max_x, max_y = p.bbox.to_xyxy()
        touches_border = (
            min_x <= margin_x
            or min_y <= margin_y
            or max_x >= image_width - margin_x
            or max_y >= image_height - margin_y
        )
        if touches_border:
            continue
        kept.append(p)
    return kept


# Returns: (result, birds, planes)

def run_detection(
    detection_model,
    image,
    slice_size,
    overlap_ratio,
    postprocess_match_threshold,
    perform_standard_pred,
    edge_margin_ratio,
    is_custom_model,
):
    """Runs SAHI on the image or RGB frame and returns (result, birds, planes)."""
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=postprocess_match_threshold,
        perform_standard_pred=perform_standard_pred,
        verbose=0,
    )

    predictions = result.object_prediction_list

    if isinstance(image, str):
        with Image.open(image) as img:
            image_width, image_height = img.size
    else:
        image_height, image_width = image.shape[:2]

    predictions = filter_border_detections(predictions, image_width, image_height, edge_margin_ratio)

    if is_custom_model:
        birds = [p for p in predictions if p.category.name.lower() == "bird"]
        planes = [p for p in predictions if p.category.name.lower() == "airplane"]
    else:
        birds = [p for p in predictions if p.category.id == COCO_BIRD_ID]
        planes = [p for p in predictions if p.category.id == COCO_AIRPLANE_ID]

    # Keep only the target classes for downstream inspection/visualization.
    result.object_prediction_list = birds + planes
    return result, birds, planes


def print_summary(label, birds, planes):
    print(f"\n== {label} ==")
    if not birds and not planes:
        print("Nie wykryto ani ptaka, ani samolotu.")
        return
    if birds:
        best = max(birds, key=lambda p: p.score.value)
        print(f"PTAK wykryty: {len(birds)}x (najlepsza pewnosc: {best.score.value:.2f})")
    if planes:
        best = max(planes, key=lambda p: p.score.value)
        print(f"SAMOLOT wykryty: {len(planes)}x (najlepsza pewnosc: {best.score.value:.2f})")


def analyze_image(
    detection_model,
    image_path,
    slice_size,
    overlap_ratio,
    is_custom_model,
    output_dir,
    postprocess_match_threshold,
    perform_standard_pred,
    edge_margin_ratio,
):
    result, birds, planes = run_detection(
        detection_model,
        image_path,
        slice_size,
        overlap_ratio,
        postprocess_match_threshold,
        perform_standard_pred,
        edge_margin_ratio,
        is_custom_model,
    )

    print_summary(os.path.basename(image_path), birds, planes)

    os.makedirs(output_dir, exist_ok=True)
    result.export_visuals(export_dir=output_dir, file_name=os.path.splitext(os.path.basename(image_path))[0])


# Returns: (x1, y1, x2, y2) of the clipped observation area on image.

def make_observation_roi(predictions, image_width, image_height, margin_factor):
    if not predictions:
        return None

    boxes = [
        (item[0] if isinstance(item, tuple) else item).bbox.to_xyxy()
        for item in predictions
    ]
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    object_width = max_x - min_x
    object_height = max_y - min_y

    min_x = max(0, int(min_x - object_width * margin_factor))
    min_y = max(0, int(min_y - object_height * margin_factor))
    max_x = min(image_width, int(max_x + object_width * margin_factor))
    max_y = min(image_height, int(max_y + object_height * margin_factor))
    return min_x, min_y, max_x, max_y
