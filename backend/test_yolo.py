from ultralytics import YOLO

print("Loading YOLO...")

model = YOLO("yolo26n.pt")

print("YOLO loaded successfully!")

results = model("https://ultralytics.com/images/bus.jpg")

print("Detection completed!")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Object: {model.names[class_id]} | "
            f"Confidence: {confidence:.2f}"
        )