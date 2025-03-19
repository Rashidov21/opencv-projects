import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

# Получаем размеры экрана
screen_width, screen_height = pyautogui.size()

# Запускаем веб-камеру
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отражаем изображение (чтобы было как в зеркале)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Конвертируем в RGB для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Координаты указательного пальца
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger.x * w), int(index_finger.y * h)

            # Переводим координаты в размеры экрана
            screen_x = np.interp(x, [0, w], [0, screen_width])
            screen_y = np.interp(y, [0, h], [0, screen_height])

            # Двигаем курсор
            pyautogui.moveTo(screen_x, screen_y, duration=0.1)

            # Выводим точку на пальце
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

    cv2.imshow("Hand Mouse Control", frame)

    # Выход по ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
