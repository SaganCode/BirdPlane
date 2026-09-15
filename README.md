# BirdPlane - Sagan Marcin
Projekt stworzony na rzech praktych w firmie SemiQa - prowadzacy Tomasz Matusiak

Projekt rozpoznaje (a przynajmniej sie stara) rozrozniac patki od samolotow na zdjeciach badz wideo (gotowe lub live)

## 1. Projekt tesowany oraz pisany przy uzyciu

- Python 3.10+
- Microsoft Windows / VS Code terminal
- Wirtualne srodowisko w katalogu .venv
- Pakiety z requirements.txt
- GPU - NVIDIA GeForce RTX 4060
- DroidCam (Klient oraz App)

Instrukcja instalacji:

    python -m venv .venv

    .\.venv\Scripts\Activate.ps1

    pip install --upgrade pip

    pip install -r requirements.txt


    Instalacja CUDA (GPU)

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    pliku model_download.py mozna uzyc do pobrania testowanych modeli

    n oraz s - nie daje rady na dalekie obiekty
    m - daje srednie wyniki
    l oraz x - najdokladniejsze ale wolniej obrabia wideo i live-wideo

## 2. Uruchamianie - wersja interaktywna:

    python detect_bird_plane.py 

    W przypadku live-wideo (stream) 

## 3. Parametry - mozna dostosowac z lini kodu, albo w pliku detect_bird_plane.py

- --model: sciezka do modelu .pt,
- --conf: poziom pewnosci detekcji,
- --slice-size: rozmiar kafelka SAHI,
- --overlap: procent nakladanie sie kafelkow,
- --image-size: rozmiar wejsciowy modelu,
- --match-threshold: threshold NMS / IoS,
- --full-image-pass: dodatkowa detekcja na calym obrazie
- --edge-margin: pomijaj detekcje przy krawedziach,
- --frame-skip: co ktora klatka analizowana w wideo, lub auto
- --max-fps: limit dla strumienia
- --observation-seconds: czas obserwacji ROI (Region of Interest) w sekundach
- --roi-video: aktywuj obserwacjacje ROI w analizie wideo
- --track-objects: aktywuj tracker obiektu
- --device: auto, cpu albo cuda:0; (domyslnie cuda-gpu)
- --out: katalog wynikow, domyslnie sahi_results

## 4. Obslugiwane pliki:

- .jpg, .jpeg, .png
- .mp4

## 5. Uwagi

- SAHI jest uzywane do podzielonej detekcji kafelkowej.
- Na obrazie detekcje dotykajajace krawedzi sa odrzucane, aby ograniczyc falszywe rozpoznania.
- W trybie ROI program zlicza ramki objiektow i drukuje podsumowanie na koncu obserwacji.
