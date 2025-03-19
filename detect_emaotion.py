import cv2
from deepface import DeepFace

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Переводим изображение в оттенки серого (для ускорения обработки)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        # Анализ эмоций с помощью DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Вывод эмоции на экран
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("Ошибка анализа:", e)

    # Отображение изображения
    cv2.imshow("Emotion Recognition", frame)

    # Выход при нажатии ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
