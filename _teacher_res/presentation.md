## Презентация

Слайд №1 - Название занятия, картинка ![](https://developers.google.com/static/mediapipe/images/solutions/examples/hand_gesture_480.png)

Слайд №2 - Задание: Проект Hands, запустите `intro.py`

Слайд №3 - Задание: Проект Hands, блокнот `hands.pynb` выполните первое задание: переделайте программу так, чтобы она работала с видеопотоком, то есть брала изображение с камеры.

Слайды №4-5 - Возможности Mediapipe c картинками отсюда https://developers.google.com/mediapipe/solutions/examples и подписями на русском (перевести)

Слайд №6 - Модули Mediapipe (таблица) с таблицей из https://developers.google.com/mediapipe/solutions/guide

Слайд №7 - Несовершенство ИИ ![](https://github.com/vv73/HandsMediapipe/raw/master/_common_res/unrecognized.png)

Слайд №8 - Задача №2. Допишите функцию, которая принимает имя картинки и определяет, есть ли рука на этой картинке

![](https://github.com/vv73/HandsMediapipe/raw/master/_common_res/task2.png)

Cлайд №9 - Решение задачи 2
```
 ...  
 detection_result = detector.detect(image)
 result = False
 if len(detection_result.handedness) > 0:
 ...
```
Слайд №10 - Подведение итогов

Что изучили, как изученное можно применить в проектах?
