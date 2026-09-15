from ultralytics import YOLO

def main():
    # Pobiera i ładuje gotowy, lekki model YOLO
    model = YOLO("yolo11s.pt")

    model.info()

    model = YOLO("yolo11n.pt")

    model.info()

    model = YOLO("yolo11l.pt")
        
    model.info()
    
    model = YOLO("yolo11m.pt")
    
    model.info()

    model = YOLO("yolo11x.pt")
        
    model.info()


    print("Model YOLO został poprawnie załadowany i jest gotowy do użycia!")

if __name__ == "__main__":
    main()