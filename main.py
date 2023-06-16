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

# 2. Подготовливаем изображение
cv_mat = cv2.cvtColor(cv2.imread("pics/hand.jpg"), cv2.COLOR_BGR2RGB)

# Перeводим его в формат Mediapipe-изображений
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

# 3. Самое главное: РАСПОЗНАЕМ
detection_result = detector.detect(image)

# Отрисовываем результат распознавания
annotated_image = draw_landmarks_on_image(image, detection_result)
cv2.imshow("Result", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Задерживаем программу до нажатия на кнопку
cv2.waitKey(0)

