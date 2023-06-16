# 0. Импортируем необходимое
# из библиотек OpenCV и MediaPipe
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# и собственную функцию для рисования
from draw_landmarks import draw_landmarks_on_image

# 1. Создаем и настраиваем детектор
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# 2 открываем видеопоток
cap = cv2.VideoCapture(0)
if not cap.isOpened():
  raise IOError("Ошибка видеозахвата!")

key = None
# завершить программу можно клавишей ESC
while cap.isOpened() and key != 27:
    # 2. Подготовливаем изображение
    ret, frame = cap.read()
    if ret != True:
        continue
    # Отражаем кадр по горизонтали
    frame = frame[:, ::-1, :]
    # Перeводим его в формат Mediapipe-изображений
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # 3. Самое главное: РАСПОЗНАЕМ
    detection_result = detector.detect(image)

    if len(detection_result.hand_landmarks) > 0:
        # нас интересует только подушечка указательного пальца (индекс 8)
        # нужно умножить координаты на размеры картинки
        x_tip = int(detection_result.hand_landmarks[0][8].x *
                    image.width)
        y_tip = int(detection_result.hand_landmarks[0][8].y *
                    image.height)
        # рисуем кружок
        cv2.circle(frame, (x_tip, y_tip), 10, (255, 0, 0), -1)
    # переводим обратно в BGR и показываем
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Example", frame)

     #Задерживаем программу до нажатия на кнопку
    key = cv2.waitKey(1)
cv2.destroyAllWindows()

