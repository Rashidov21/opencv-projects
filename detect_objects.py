import cv2
from ultralytics import YOLO

# Загружаем модель YOLOv8 (предобученную)
model = YOLO("yolov8n.pt")

# Запускаем камеру
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем YOLO для детекции объектов
    results = model(frame)

    # Отображаем найденные объекты
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамки
            confidence = box.conf[0]  # Уверенность модели
            label = result.names[int(box.cls[0])]  # Название объекта

            # Рисуем рамку и подпись
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Показываем изображение
    cv2.imshow("Object Detection", frame)

    # Выход при нажатии ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
