import cv2


def draw_detections_on_frame(frame_bgr, birds, planes, x_offset, y_offset):
    for prediction, object_id in birds + planes:
        label = "bird" if prediction.category.name.lower() == "bird" else "airplane"
        color = (0, 255, 255) if label == "bird" else (0, 0, 255)
        min_x, min_y, max_x, max_y = (int(v) for v in prediction.bbox.to_xyxy())
        min_x += x_offset
        max_x += x_offset
        min_y += y_offset
        max_y += y_offset
        cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), color, 2)
        label_text = f"{label} {prediction.score.value:.2f}"
        if object_id is not None:
            label_text = f"{label} #{object_id} {prediction.score.value:.2f}"
        cv2.putText(
            frame_bgr,
            label_text,
            (min_x, max(min_y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame_bgr
