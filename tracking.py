def detection_iou(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def detections_match(first_box, second_box, iou_threshold):
    first_x1, first_y1, first_x2, first_y2 = first_box
    second_x1, second_y1, second_x2, second_y2 = second_box
    gap = max(
        4,
        min(first_x2 - first_x1, first_y2 - first_y1,
            second_x2 - second_x1, second_y2 - second_y1) * 0.15,
    )
    separated_horizontally = first_x2 + gap < second_x1 or second_x2 + gap < first_x1
    separated_vertically = first_y2 + gap < second_y1 or second_y2 + gap < first_y1
    return detection_iou(first_box, second_box) >= iou_threshold or not (
        separated_horizontally or separated_vertically
    )


class ObjectTracker:
    def __init__(self, match_threshold=0.1, duplicate_threshold=0.3, max_missed=10):
        self.match_threshold = match_threshold
        self.duplicate_threshold = duplicate_threshold
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    def reset(self):
        self.tracks.clear()

    def update(self, predictions, x_offset=0, y_offset=0):
        unique_predictions = []
        for prediction in sorted(predictions, key=lambda item: item.score.value, reverse=True):
            box = prediction.bbox.to_xyxy()
            full_box = (
                box[0] + x_offset,
                box[1] + y_offset,
                box[2] + x_offset,
                box[3] + y_offset,
            )
            category = prediction.category.name.lower()
            if any(
                category == other.category.name.lower()
                and detections_match(full_box, other_box, self.duplicate_threshold)
                for other, other_box in unique_predictions
            ):
                continue
            unique_predictions.append((prediction, full_box))

        for track in self.tracks.values():
            track["missed"] += 1

        tracked = []
        used_track_ids = set()
        for prediction, full_box in unique_predictions:
            category = prediction.category.name.lower()
            best_id = None
            best_iou = self.match_threshold
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids or track["category"] != category:
                    continue
                overlap = detection_iou(full_box, track["box"])
                if overlap >= best_iou or detections_match(full_box, track["box"], 0.0):
                    best_id = track_id
                    best_iou = overlap

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = {"category": category, "box": full_box, "missed": 0}
            used_track_ids.add(best_id)
            tracked.append((prediction, best_id))

        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] <= self.max_missed
        }
        return tracked
