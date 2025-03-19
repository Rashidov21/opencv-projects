import cv2
import mediapipe as mp

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Отражаем изображение, чтобы было как в зеркале
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Координаты контрольных точек пальцев
            landmarks = hand_landmarks.landmark
            fingers = [4, 8, 12, 16, 20]  # Кончики пальцев (большой, указательный, средний, безымянный, мизинец)

            count = 0
            for i in range(1, 5):  # Проверяем 4 пальца (кроме большого)
                if landmarks[fingers[i]].y < landmarks[fingers[i] - 2].y:
                    count += 1
            
            # Проверяем большой палец отдельно (по горизонтали)
            if landmarks[fingers[0]].x > landmarks[fingers[0] - 1].x:
                count += 1

            # Выводим количество пальцев
            cv2.putText(frame, f"Fingers: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Выход по ESC
        break

cap.release()
cv2.destroyAllWindows()
