import argparse
import os
import sys
import torch

from config import DEFAULT_MODEL_PATH
from detection import analyze_image
from media import collect_media_paths, is_stream_source, prompt_for_source
from processing import analyze_stream, analyze_video


def main():
    parser = argparse.ArgumentParser(
        description="Wykrywa ptaki i samoloty na zdjeciach przy uzyciu YOLO + SAHI."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Sciezka do zdjecia, wideo, folderu z nimi lub URL strumienia (jesli pominiete, program zapyta o zrodlo)",
    )
    parser.add_argument("--model", default=None, help="Sciezka do wlasnego pliku .pt")
    parser.add_argument("--conf", type=float, default=0.3, help="Prog pewnosci detekcji")
    parser.add_argument("--slice-size", type=int, default=512, help="Rozmiar kafelka SAHI (px) - mniejszy = lepiej widac male, odlegle obiekty")
    parser.add_argument("--overlap", type=float, default=0.35, help="Nakladanie sie kafelkow (0-1)")
    parser.add_argument("--image-size", type=int, default=640, help="Rozmiar do jakiego skalowany jest kazdy kafelek przed detekcja (wiekszy = drobniejsze obiekty lepiej widoczne)")
    parser.add_argument("--match-threshold", type=float, default=0.5, help="Prog scalania nakladajacych sie detekcji miedzy kafelkami")
    parser.add_argument("--full-image-pass", action="store_true", help="Dodatkowo wykonaj detekcje na calym obrazie (pomaga na blisko polozonych obiektach ucinanych przez kafelki)")
    parser.add_argument("--edge-margin", type=float, default=0.02, help="Szerokosc strefy przy krawedzi zdjecia (jako ulamek wymiaru) - detekcje w niej sa calkowicie odrzucane")
    parser.add_argument("--frame-skip", type=float, default=1.0, help="Analizuj co N-ta klatke wideo; 'auto' dobiera wartosc do rozdzielczosci, 1 analizuje kazda klatke")
    parser.add_argument("--max-fps", type=float, default=24.0, help="Maksymalna liczba detekcji na sekunde dla zywego strumienia (0 = bez limitu, kazda klatka)")
    parser.add_argument("--observation-seconds", type=float, default=2.0, help="Czas obserwacji obszaru po wykryciu obiektu")
    parser.add_argument("--roi-video", action="store_true", help="Uzyj ROI i glosowania przez observation-seconds dla plikow wideo")
    parser.add_argument("--track-objects", action=argparse.BooleanOptionalAction, default=True, help="Wlacz lub wylacz ID tracker")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' lub 'cuda:0'")
    parser.add_argument("--out", default="sahi_results", help="Folder na zdjecia/wideo z zaznaczonymi detekcjami")
    args = parser.parse_args()

    if not args.input:
        args.input, prompted_roi, prompted_tracker = prompt_for_source()
        args.roi_video = prompted_roi
        args.track_objects = prompted_tracker

    model_path = args.model or DEFAULT_MODEL_PATH
    is_custom_model = bool(args.model)
    if not os.path.isfile(model_path):
        print(f"Nie znaleziono modelu: {model_path}")
        sys.exit(1)

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Uzywany model: {model_path} ({'wlasny' if is_custom_model else 'ogolny COCO'})")
    print(f"Uzywane urzadzenie: {device}")

    from sahi import AutoDetectionModel

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=args.conf,
        image_size=args.image_size,
        device=device,
    )

    if is_stream_source(args.input):
        analyze_stream(
            detection_model,
            args.input,
            args.slice_size,
            args.overlap,
            is_custom_model,
            args.match_threshold,
            args.full_image_pass,
            args.edge_margin,
            args.max_fps,
            args.observation_seconds,
            args.track_objects,
        )
        return

    image_paths, video_paths = collect_media_paths(args.input)

    if not image_paths and not video_paths:
        print(f"Brak zdjec ani wideo do przetworzenia w: {args.input}")
        return

    for image_path in image_paths:
        analyze_image(
            detection_model,
            image_path,
            args.slice_size,
            args.overlap,
            is_custom_model,
            args.out,
            args.match_threshold,
            args.full_image_pass,
            args.edge_margin,
        )

    for video_path in video_paths:
        analyze_video(
            detection_model,
            video_path,
            args.slice_size,
            args.overlap,
            is_custom_model,
            args.out,
            args.match_threshold,
            args.full_image_pass,
            args.edge_margin,
            args.frame_skip,
            args.observation_seconds,
            args.roi_video,
            args.track_objects,
        )

    print(f"\nGotowe. Wyniki z zaznaczonymi detekcjami zapisano w: {args.out}")


if __name__ == "__main__":
    main()
