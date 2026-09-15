import threading
from pathlib import Path

import cv2

from config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


# Returns: bool

def is_stream_source(input_path):
    """Recognizes a live stream: URL such as DroidCam or a direct stream endpoint."""
    return input_path.startswith("http://") or input_path.startswith("https://")


# Returns: (image_paths, video_paths)

def collect_media_paths(input_path):
    """Returns (image_paths, video_paths) found under the given input path."""
    path = Path(input_path)
    if path.is_dir():
        images = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        videos = sorted(str(p) for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        return images, videos

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return [], [str(path)]
        return [str(path)], []

    raise FileNotFoundError(f"Nie znaleziono pliku ani folderu: {input_path}")


# Returns: (cap, used_source) or (None, source)

def open_stream_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap, source
    cap.release()
    return None, source


class LatestFrameReader:
    """Reads frames in a background thread and keeps the newest frame available.

    This prevents the stream from lagging behind the camera when SAHI inference
    is slower than the incoming frame rate.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = True
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                with self.lock:
                    self.ok = False
                return
            with self.lock:
                self.frame = frame

    def read(self):
        # Returns: (ok, frame) frame is None until the first capture arrives.
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)


# Returns: (entered source text, roi flag for video, tracker flag)

def prompt_for_source():
    """Prompts for the source when none is supplied as a command-line argument."""
    print("Wybierz zrodlo danych:")
    print("  1) Plik lub folder (zdjecia/wideo), zwykly tryb")
    print("  2) Zywy strumien (np. DroidCam - URL)")
    print("  3) Plik lub folder z wideo, tryb ROI")
    choice = input("Twoj wybor (1/2/3): ").strip()

    if choice == "2":
        source = input("Podaj URL strumienia (np. http://192.168.0.10:4747/video): ").strip()
        roi = False
    else:
        source = input("Podaj sciezke do zdjecia, wideo lub folderu: ").strip()
        roi = choice == "3"

    tracker = input("Czy wlaczyc ID tracker? (t/n): ").strip().lower() in ("t", "tak", "y", "yes")
    return source, roi, tracker
