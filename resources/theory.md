# Mediapipe. Модуль Hands

## Чему научимся

На этом занятии мы:
* Познакомимся с библиотекой Mediapipe
* Научитесь использовать модуль Hands - распознавать положение кистей рук 

Эти знания можно будет использовать для проектов с управлением программой жестами руки.

## Введение

Сразу пример (Проект `Hands`, скрипт `main.py`):

```python
1: # 0. Импортируем необходимое
2: # из библиотек OpenCV и MediaPipe
3: import cv2
4: import mediapipe as mp
5: from mediapipe.tasks import python
6: from mediapipe.tasks.python import vision
7: # и собственную функцию для рисования
8: from draw_landmarks import draw_landmarks_on_image
9: 
10: # 1. Создаем и настраиваем детектор
11: base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
12: options = vision.HandLandmarkerOptions(base_options=base_options,
13:                                        num_hands=2)
14: detector = vision.HandLandmarker.create_from_options(options)
15: 
16: # 2. Подготовливаем изображение
17: cv_mat = cv2.cvtColor(cv2.imread("pics/hand.jpg"), cv2.COLOR_BGR2RGB)
18: 
19: # Перводим его в формат Mediapipe-изображений
20: image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
21: 
22: # 3. Самое главное: РАСПОЗНАЕМ
23: detection_result = detector.detect(image)
24: 
25: # Отрисовываем результат распознавания
26: annotated_image = draw_landmarks_on_image(image, detection_result)
27: cv2.imshow("Result", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
28: 
29: # Задерживаем программу до нажатия на кнопку
30: cv2.waitKey(0)
```
Запустим эту программу, получим окно с изображением руки, на которое наложены ключевые точки слово "Left" - детектировалась левая рука.

![](https://github.com/vv73/HandsMediapipe/raw/master/_common_res/annotated_hand.png)

В этой программе много строк, но они просты для понимания, если не вдаваться в детали.

Наиболее сложные строчки, наверное, *11-13*
```python
11: base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
12: options = vision.HandLandmarkerOptions(base_options=base_options,
13:                                        num_hands=2)
```
Это настройки (стандартные, которым передается модель, которая и распознает ладони и определяет их ключевые точки) по которым строится в строке 14 объект-распознаватель.

Функция `draw_landmarks_on_image()` в 14 строке вынесена в отдельный файл, она не очень сложная, но объемная.

# Что такое MediaPipe и зачем

Ссылка на сайт библиотеки:  [](https://developers.google.com/mediapipe). 

**MediaPipe** для нас - это средство детектирования рук, поз, мимики лица с очень простым интерфейсом и высокой скоростью работы. Библиотека MediaPipe разработана компанией Google, поэтому самые богатые возможности у Android-версии. Однако, много возможностей доступно и для программирования на Python.

Сейчас MediaPipe поддерживает не только распознавание изображений, но умеет работать с текстом и аудио. Причем, на настоящий момент все возможности поддерживаются и на Python и в Android на Java и Kotlin и на Web-страницах с использованием языка Java Script.

**Яна, Вот здесь надо вставить табличку с https://developers.google.com/mediapipe/solutions/guide**

Внутри MediaPipe используются нейронные сети, но мы сейчас пользуемся возможностями, не вникая во внутреннее устройство и ограничимся готовыми моделями. То есть, собственно, распознавание, например, руки для нас будет "магическим", а наша программистская задача будет состоять в обработке полученных данных, координат ключевых точек кисти. На этом занятии мы изучим только работу с модулем Нands, на одном из следующих поработаем c модулем Pose, a проект можно делать с использованием любого модуля Mediapipe, в работе с другими модулями практически не будет отличий.

# Разберемся с модулем Hands!

В программе `main.py` добавим вывод результата в виде текста:

...
**print(detection_result)**
cv2.waitKey(0)

Получим 
```
HandLandmarkerResult(handedness=[[Category(index=0, score=0.947941780090332, display_name='Left', category_name='Left')]], hand_landmarks=[[NormalizedLandmark(x=0.3054841458797455, y=0.6554181575775146, z=6.081149876990821e-07, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.29667195677757263, y=0.5337704420089722, z=-0.04708409309387207, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.33271777629852295, y=0.4215508699417114, z=-0.09257516264915466, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3721213936805725, y=0.32396388053894043, z=-0.13034261763095856, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3804454505443573, y=0.23811903595924377, z=-0.17078007757663727, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.48494574427604675, y=0.46120521426200867, z=-0.11689281463623047, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.5957748293876648, y=0.3660905361175537, z=-0.1651155650615692, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.6649013161659241, y=0.3077567219734192, z=-0.1930573582649231, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7268192172050476, y=0.262215256690979, z=-0.21168960630893707, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.5292543172836304, y=0.5301592350006104, z=-0.11824683845043182, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.6610936522483826, y=0.4433966875076294, z=-0.17202512919902802, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7518598437309265, y=0.3876284658908844, z=-0.20369577407836914, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8291220664978027, y=0.3465093970298767, z=-0.2244330197572708, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.548652708530426, y=0.5999841094017029, z=-0.1183793917298317, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.6858201622962952, y=0.5530462265014648, z=-0.16976428031921387, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7755565643310547, y=0.5231123566627502, z=-0.20256371796131134, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.8419668674468994, y=0.4981439411640167, z=-0.22032372653484344, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.5542590618133545, y=0.664034366607666, z=-0.11851301789283752, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.6700118780136108, y=0.6497011780738831, z=-0.1591109335422516, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7401710748672485, y=0.6338604092597961, z=-0.1782578080892563, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.7961540818214417, y=0.6172436475753784, z=-0.1880868524312973, visibility=0.0, presence=0.0)]], hand_world_landmarks=[[Landmark(x=-0.04677943140268326, y=0.02591538056731224, z=0.04326263815164566, visibility=0.0, presence=0.0), Landmark(x=-0.04082027077674866, y=-0.0012519191950559616, z=0.03517264127731323, visibility=0.0, presence=0.0), Landmark(x=-0.03228621557354927, y=-0.01599077321588993, z=0.01855815201997757, visibility=0.0, presence=0.0), Landmark(x=-0.02439483068883419, y=-0.03413946181535721, z=-4.903378794551827e-05, visibility=0.0, presence=0.0), Landmark(x=-0.025819897651672363, y=-0.06068330258131027, z=-0.018411699682474136, visibility=0.0, presence=0.0), Landmark(x=-0.0014992373762652278, y=-0.011522599495947361, z=-0.0015358274104073644, visibility=0.0, presence=0.0), Landmark(x=0.011138757690787315, y=-0.02661372348666191, z=-0.005821044556796551, visibility=0.0, presence=0.0), Landmark(x=0.02372930757701397, y=-0.0387333482503891, z=-0.0009090003441087902, visibility=0.0, presence=0.0), Landmark(x=0.03883671760559082, y=-0.055798232555389404, z=0.0061420113779604435, visibility=0.0, presence=0.0), Landmark(x=9.748339653015137e-05, y=-0.0018816484371200204, z=-0.001332716317847371, visibility=0.0, presence=0.0), Landmark(x=0.0202338807284832, y=-0.011569801717996597, z=-0.007718841545283794, visibility=0.0, presence=0.0), Landmark(x=0.040813956409692764, y=-0.029905280098319054, z=-0.0066446722485125065, visibility=0.0, presence=0.0), Landmark(x=0.06388561427593231, y=-0.04822714626789093, z=0.0027109181974083185, visibility=0.0, presence=0.0), Landmark(x=0.0036585950292646885, y=0.006129223853349686, z=-0.0008797055925242603, visibility=0.0, presence=0.0), Landmark(x=0.01935535855591297, y=0.0041543543338775635, z=-0.0024172181729227304, visibility=0.0, presence=0.0), Landmark(x=0.043495483696460724, y=-0.0024893246591091156, z=0.0015347398584708571, visibility=0.0, presence=0.0), Landmark(x=0.06608940660953522, y=-0.010304881259799004, z=0.01083249319344759, visibility=0.0, presence=0.0), Landmark(x=-0.002135457471013069, y=0.012784033082425594, z=0.009981910698115826, visibility=0.0, presence=0.0), Landmark(x=0.0176137313246727, y=0.018632013350725174, z=0.0068219746463000774, visibility=0.0, presence=0.0), Landmark(x=0.03819430246949196, y=0.01870032772421837, z=0.005857598502188921, visibility=0.0, presence=0.0), Landmark(x=0.05166883021593094, y=0.010128375142812729, z=0.011611340567469597, visibility=0.0, presence=0.0)]])
```
Вывод выглядит устрашающим, но это только на первый взгляд.

Это объект. Сколько в нем подобъектов? Всего три: `handedness`, `hand_landmarks`, и  ` hand_world_landmarks`. Это информация о руках (в нашем выводе обнаружилась левая), и о контрольных точках. 

`handedness` - дает координаты относительно картинки, например, x = 0.5, y = 0.5 - означает, что точка в центре изображения

`hand_landmarks` - дает ту же информацию, но не относительно картинки, а в реальном мире, в метрах, при этом точкой (0, 0, 0) считается точка запястья. Интересно, что детектор определяет и третью координату z. Координата z представляет глубину контрольной точки, при этом глубина на запястье является исходной точкой. Чем меньше значение, тем ближе точка к камере. Величина z использует примерно тот же масштаб, что и x.

Можно вывести только что-нибудь одно, например, только информацию о руках:

`print(detection_result.handedness)`

Получим 

```[[Category(index=0, score=0.947941780090332, display_name='Left', category_name='Left')]]```

Левая рука определилась с вероятностью 95%.

## Ключевые точки

Индексы точек - самое важное!

Индексы расположены в порядке как на картинке

![](https://github.com/vv73/HandsMediapipe/raw/master/_common_res/indexes.png)

Перепишем программу для видео и будем следить только за указательным пальцем. Верхняя фаланга большого пальца имеет индекс 8.

```python
 ...
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
...
```

![](https://github.com/vv73/HandsMediapipe/raw/master/_common_res/mp.gif)

Полный код находится в файле `example.py`. 

# Итоги занятия

На этом занятии мы научились работать с модулем Hands библиотеки Mediapipe, а именно:

* Понимать, есть ли на изображении руки и сколько их
* Следить за определенными частями ладони 
