import os
import time

import cv2
import numpy as np

from detection import run_detection, make_observation_roi, print_summary
from drawing import draw_detections_on_frame
from media import LatestFrameReader, open_stream_capture
from tracking import ObjectTracker


def print_observation_counts_and_winner(bird_flags, plane_flags):
    print(f"Wynik obserwacji: boxy ptakow={bird_flags}, boxy samolotow={plane_flags}")
    if bird_flags > plane_flags:
        print(f"Wniosek: wiecej boxow ptakow ({bird_flags}) niz boxow samolotow ({plane_flags}) -> najprawdopodobniej PTAK.")
    elif plane_flags > bird_flags:
        print(f"Wniosek: wiecej boxow samolotow ({plane_flags}) niz boxow ptakow ({bird_flags}) -> najprawdopodobniej SAMOLOT.")
    else:
        print("Wniosek: brak rozstrzygniecia - liczba boxow ptakow i samolotow jest rowna.")


# Video processing

def analyze_video(
    detection_model,
    video_path,
    slice_size,
    overlap_ratio,
    is_custom_model,
    output_dir,
    postprocess_match_threshold,
    perform_standard_pred,
    edge_margin_ratio,
    frame_skip,
    observation_seconds,
    roi_observation,
    tracker_enabled,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie mozna otworzyc wideo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_skip == "auto":
        frame_skip = max(1, round((width * height) / (1280 * 720)))
    else:
        frame_skip = int(frame_skip)
        if frame_skip < 1:
            raise ValueError("frame_skip musi byc liczba >= 1 albo wartoscia 'auto'")

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_detected.mp4"
    out_path = os.path.join(output_dir, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    frames_with_bird = 0
    frames_with_plane = 0
    best_bird_conf = 0.0
    best_plane_conf = 0.0
    roi = None
    observation_start_frame = None
    birds, planes = [], []
    roi_x, roi_y = 0, 0
    tracker = ObjectTracker()
    bird_ids = set()
    plane_ids = set()
    bird_hits = 0
    plane_hits = 0
    bird_flags = 0
    plane_flags = 0
    roi_bird_seen = False
    roi_plane_seen = False

    print(f"\n== {os.path.basename(video_path)} ==")
    print(f"Rozdzielczosc: {width}x{height}, frame-skip: {frame_skip}")
    start_time = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip == 0:
            if roi_observation and roi is not None:
                x1, y1, x2, y2 = roi
                detection_frame = frame[y1:y2, x1:x2]
                roi_x, roi_y = x1, y1
                detection_edge_margin = 0
            else:
                detection_frame = frame
                roi_x, roi_y = 0, 0
                detection_edge_margin = edge_margin_ratio

            rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            _, birds, planes = run_detection(
                detection_model,
                rgb_frame,
                slice_size,
                overlap_ratio,
                postprocess_match_threshold,
                perform_standard_pred,
                detection_edge_margin,
                is_custom_model,
            )
            tracked_objects = tracker.update(birds + planes, roi_x, roi_y) if tracker_enabled else [(p, None) for p in birds + planes]
            tracked_birds = [item for item in tracked_objects if item[0].category.name.lower() == "bird"]
            tracked_planes = [item for item in tracked_objects if item[0].category.name.lower() == "airplane"]
            detected_objects = tracked_birds + tracked_planes
            if roi_observation:
                if roi is None and detected_objects:
                    roi = make_observation_roi(detected_objects, width, height, 2.5)
                    observation_start_frame = frame_idx
                    roi_bird_seen = bool(birds)
                    roi_plane_seen = bool(planes)
                    if tracker_enabled:
                        bird_ids = {object_id for _, object_id in tracked_birds}
                        plane_ids = {object_id for _, object_id in tracked_planes}
                    bird_flags = len(tracked_birds)
                    plane_flags = len(tracked_planes)
                    print(f"Obiekt wykryty - obserwuje obszar przez {observation_seconds:.1f} s.")
                elif roi is not None:
                    roi_bird_seen = roi_bird_seen or bool(birds)
                    roi_plane_seen = roi_plane_seen or bool(planes)
                    if tracker_enabled:
                        bird_ids.update(object_id for _, object_id in tracked_birds)
                        plane_ids.update(object_id for _, object_id in tracked_planes)
                    bird_flags += len(tracked_birds)
                    plane_flags += len(tracked_planes)
                    if detected_objects:
                        local_roi = make_observation_roi(
                            detected_objects, detection_frame.shape[1], detection_frame.shape[0], 2.5
                        )
                        roi = (
                            max(0, local_roi[0] + roi_x),
                            max(0, local_roi[1] + roi_y),
                            min(width, local_roi[2] + roi_x),
                            min(height, local_roi[3] + roi_y),
                        )

                if roi is not None and frame_idx - observation_start_frame >= round(observation_seconds * fps):
                    print_observation_counts_and_winner(bird_flags, plane_flags)
                    roi = None
                    observation_start_frame = None
                    birds, planes = [], []
                    bird_ids.clear()
                    plane_ids.clear()
                    bird_hits = 0
                    plane_hits = 0
                    bird_flags = 0
                    plane_flags = 0
                    roi_bird_seen = False
                    roi_plane_seen = False
                    tracker.reset()
            else:
                if tracked_birds:
                    frames_with_bird += 1
                    best_bird_conf = max(best_bird_conf, max(p.score.value for p in birds))
                if tracked_planes:
                    frames_with_plane += 1
                    best_plane_conf = max(best_plane_conf, max(p.score.value for p in planes))

            frame = draw_detections_on_frame(frame, tracked_birds, tracked_planes, roi_x, roi_y)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Przetworzono {frame_idx} klatek...")

    cap.release()
    writer.release()

    elapsed = time.monotonic() - start_time
    processing_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Czas przetwarzania: {elapsed:.1f}s dla {frame_idx} klatek ({processing_fps:.1f} klatek/s)")
    print(f"Klatek z ptakiem: {frames_with_bird} (najlepsza pewnosc: {best_bird_conf:.2f})")
    print(f"Klatek z samolotem: {frames_with_plane} (najlepsza pewnosc: {best_plane_conf:.2f})")
    print(f"Zapisano wideo z detekcjami: {out_path}")


# Stream processing

def analyze_stream(
    detection_model,
    source,
    slice_size,
    overlap_ratio,
    is_custom_model,
    postprocess_match_threshold,
    perform_standard_pred,
    edge_margin_ratio,
    max_fps,
    observation_seconds,
    tracker_enabled,
):
    """Live preview for a stream such as DroidCam: stream URL."""
    cap, source = open_stream_capture(source)
    if cap is None:
        print(f"Nie mozna otworzyc strumienia: {source}")
        print("Sprawdz: czy telefon i komputer sa w tej samej sieci WiFi, czy adres/port sa poprawne (DroidCam dziala po http, nie https), oraz czy aplikacja DroidCam jest uruchomiona.")
        return

    print(f"Podlaczono do strumienia: {source} (wyjscie: klawisz 'q')")
    reader = LatestFrameReader(cap)
    birds, planes = [], []
    roi = None
    observation_started = None
    tracked_birds, tracked_planes = [], []
    bird_ids = set()
    plane_ids = set()
    bird_flags = 0
    plane_flags = 0
    roi_bird_seen = False
    roi_plane_seen = False
    tracker = ObjectTracker()
    bird_ids = set()
    plane_ids = set()
    min_detect_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    last_detect_time = 0.0
    stream_frame_idx = 0
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                print("Utracono polaczenie ze strumieniem.")
                break
            if frame is None:
                continue

            now = time.monotonic()
            if now - last_detect_time >= min_detect_interval:
                if roi is None:
                    detection_frame = frame
                    roi_x, roi_y = 0, 0
                    detection_edge_margin = edge_margin_ratio
                else:
                    x1, y1, x2, y2 = roi
                    detection_frame = frame[y1:y2, x1:x2]
                    roi_x, roi_y = x1, y1
                    detection_edge_margin = 0

                rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                _, birds, planes = run_detection(
                    detection_model,
                    rgb_frame,
                    slice_size,
                    overlap_ratio,
                    postprocess_match_threshold,
                    perform_standard_pred,
                    detection_edge_margin,
                    is_custom_model,
                )
                last_detect_time = now

                tracked_objects = tracker.update(birds + planes, roi_x, roi_y) if tracker_enabled else [(p, None) for p in birds + planes]
                tracked_birds = [item for item in tracked_objects if item[0].category.name.lower() == "bird"]
                tracked_planes = [item for item in tracked_objects if item[0].category.name.lower() == "airplane"]
                detected_objects = tracked_birds + tracked_planes
                if roi is None and detected_objects:
                    roi = make_observation_roi(
                        detected_objects, frame.shape[1], frame.shape[0], 2.5
                    )
                    observation_started = now
                    roi_bird_seen = bool(birds)
                    roi_plane_seen = bool(planes)
                    if tracker_enabled:
                        bird_ids = {object_id for _, object_id in tracked_birds}
                        plane_ids = {object_id for _, object_id in tracked_planes}
                    bird_flags = len(tracked_birds)
                    plane_flags = len(tracked_planes)
                    print("Obiekt wykryty - obserwuje powiekszony obszar przez "
                          f"{observation_seconds:.1f} s.")
                elif roi is not None and detected_objects:
                    roi_bird_seen = roi_bird_seen or bool(birds)
                    roi_plane_seen = roi_plane_seen or bool(planes)
                    if tracker_enabled:
                        bird_ids.update(object_id for _, object_id in tracked_birds)
                        plane_ids.update(object_id for _, object_id in tracked_planes)
                    bird_flags += len(tracked_birds)
                    plane_flags += len(tracked_planes)
                    local_roi = make_observation_roi(
                        detected_objects, frame.shape[1], frame.shape[0], 2.5
                    )
                    roi = (
                        max(0, local_roi[0] + roi_x),
                        max(0, local_roi[1] + roi_y),
                        min(frame.shape[1], local_roi[2] + roi_x),
                        min(frame.shape[0], local_roi[3] + roi_y),
                    )

                if roi is not None and now - observation_started >= observation_seconds:
                    print_observation_counts_and_winner(bird_flags, plane_flags)
                    roi = None
                    observation_started = None
                    birds, planes = [], []
                    bird_ids.clear()
                    plane_ids.clear()
                    bird_hits = 0
                    plane_hits = 0
                    bird_flags = 0
                    plane_flags = 0
                    roi_bird_seen = False
                    roi_plane_seen = False
                    tracker.reset()

                    frame = draw_detections_on_frame(frame, tracked_birds, tracked_planes, roi_x if roi else 0, roi_y if roi else 0)

            stream_frame_idx += 1
            cv2.imshow("BirdPlane - DroidCam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()

        print(f"\nPodsumowanie: flagi ptak={bird_flags}, samolot={plane_flags}")
        if not bird_flags and not plane_flags:
            print("Wniosek: w trakcie sesji nie wykryto ani ptaka, ani samolotu.")
        elif bird_flags > plane_flags:
            print("Wniosek: to najprawdopodobniej PTAK.")
        elif plane_flags > bird_flags:
            print("Wniosek: to najprawdopodobniej SAMOLOT.")
        else:
            print("Wniosek: remis - tyle samo wykryc ptaka i samolotu.")
