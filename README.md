import pygame
import random
import cv2
import mediapipe as mp
import numpy as np

def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

# основные цвета, которые я буду использовать
cveta = [
    (0, 0, 0),  # Черный
    (120, 37, 179),  # Фиолетовый
    (100, 179, 179),  # Голубой
    (80, 34, 22),  # Коричневый
    (80, 134, 22),  # Зеленый
    (180, 34, 22),  # Красный
    (180, 34, 122),  # Розовый
]

# Тут придется попотеть, потому что чтобы создать фигуру, я хочу сделать отдельный класс
# Этот лайфхак я подсмотрела на зарубежных сайтах, но данный код мой
class Figura:
    x = 0
    y = 0

    figury = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],  # Прямая полоска
        [[4, 5, 9, 10], [2, 6, 5, 9]],  # Z
        [[6, 7, 9, 10], [1, 5, 6, 10]],  # Обратная Z
        [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],  # L
        [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],  # Обратная L
        [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],  # T
        [[1, 2, 5, 6]],  # Квадратик
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tip = random.randint(0, len(self.figury) - 1)  # Выбираем из нашего списка фигур рандомную
        self.cveta = random.randint(1, len(cveta) - 1)  # И выбираем ей случайный цвет
        self.povorot = 0  # Сохраняем текущий поворот

    # Прописываем геттер
    def izobrazhenie(self):
        return self.figury[self.tip][self.povorot]

    # Вращение фигуры
    def povernut(self):
        self.povorot = (self.povorot + 1) % len(self.figury[self.tip])

# Непосредственно класс, который управляет самой игрой.
# Поскольку недавно мы проходили классы, это наилучший из вариантов
class Tetris:
    def __init__(self, vysota, shirina):
        self.uroven = 2  # Уровень сложности
        self.schet = 0  # Наши считаемые баллы
        self.sostoyanie = "start"  # Работает игра сейчас или нет, в каком она находится состоянии
        self.pole = []  # Игровое поле
        self.vysota = 0
        self.shirina = 0
        self.x = 100  # Смещение игрового поля по x
        self.y = 60  # Смещение игрового поля по y
        self.uvelichenie = 20  # Размер клетки
        self.figura = None  # Текущая фигура

        self.vysota = vysota
        self.shirina = shirina
        self.pole = []
        self.schet = 0
        self.sostoyanie = "start"

        for i in range(vysota):
            new_line = []
            for j in range(shirina):
                new_line.append(0)
            self.pole.append(new_line)

    # Создаем новую фигру
    def novaya_figura(self):
        self.figura = Figura(3, 0)

    # Проверяем, что они не пересекаются
    def peresechenie(self):
        peresechenia = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figura.izobrazhenie():
                    if i + self.figura.y > self.vysota - 1 or \
                            j + self.figura.x > self.shirina - 1 or \
                            j + self.figura.x < 0 or \
                            self.pole[i + self.figura.y][j + self.figura.x] > 0:
                        peresechenia = True
        return peresechenia

    # удаляем строку
    def break_lines(self):
        lines = 0
        for i in range(1, self.vysota):
            zeros = 0
            for j in range(self.shirina):
                if self.pole[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.shirina):
                        self.pole[i1][j] = self.pole[i1 - 1][j]
        self.schet += lines ** 2


    def go_space(self):
        while not self.peresechenie():
            self.figurа.y += 1
        self.figurа.y -= 1
        self.fiksatsiya()


    # Спускаем фигру вниз
    def v_niz(self):
        self.figura.y += 1
        if self.peresechenie():
            self.figura.y -= 1
            self.fiksatsiya()

    # Фиксируем фигуру на поле после касания пола или другой фигуры снизу
    def fiksatsiya(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figura.izobrazhenie():
                    self.pole[i + self.figura.y][j + self.figura.x] = self.figura.cveta
        self.break_lines()
        self.novaya_figura()
        if self.peresechenie():
            self.sostoyanie = "gameover"

    # Двигаем в бок с помощью кулачков
    def v_bok(self, dx):
        star_x = self.figura.x
        self.figura.x += dx
        if self.peresechenie():
            self.figura.x = star_x

    # Поворачиваем фигуру
    def povernut(self):
        star_povorot = self.figura.povorot
        self.figura.povernut()
        if self.peresechenie():
            self.figura.povorot = star_povorot

# Инициализация Mediapipe и OpenCV
handsDetector = mp.solutions.hands.Hands()
# ruki = handsDetector.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Инициализация Pygame
pygame.init()

# # Инициализация расслабляющей музыки
# pygame.mixer.music.load("music.mp3")
# pygame.mixer.music.play()

# Определение цветов
CHERNYY = (0, 0, 0)
BELYI = (255, 255, 255)
SERYI = (128, 128, 128)

# Размеры экрана
razmer = (400, 500)
ekran = pygame.display.set_mode(razmer)
pygame.display.set_caption("Тетрисчк")

# Моя камера ура
cap = cv2.VideoCapture(0)

konets = False
chas = pygame.time.Clock()
fps = 25
igra = Tetris(20, 10)
igra.novaya_figura()

# Переменные для кулачков, которые будут двигать фигурку
pred_fist_pravaya = False
pred_fist_levaya = False
schetchik = 0


# Сама игра
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Получаем точки и параметры руки
            points = get_points(hand_landmarks.landmark, flippedRGB.shape)
            (x, y), r = cv2.minEnclosingCircle(points)
            ws = palm_size(hand_landmarks.landmark, flippedRGB.shape)

            # Рисуем контуры
            cv2.drawContours(flippedRGB, [points], 0, (255, 0, 0), 2)

            # Проверяем соотношение размера окружности к размеру ладони
            if 2 * r / ws > 1.3:
                cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 0, 255), 2)
                # Кулак разжат
                if hand_idx == 0:
                    prev_fist_left = False
                else:
                    prev_fist_right = False
            else:
                cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
                if hand_idx == 0 and not prev_fist_left:
                    # Левый кулак был сжат
                    igra.v_bok(-1)
                    prev_fist_left = True
                elif hand_idx == 1 and not prev_fist_right:
                    # Правый кулак был сжат
                    igra.v_bok(1)
                    prev_fist_right = True

    # Движение вниз
    schetchik += 1
    if schetchik % (fps // igra.uroven) == 0 and igra.sostoyanie == "start":
        igra.v_niz()

    # Проверка чтобы выходил
    for sobitie in pygame.event.get():
        if sobitie.type == pygame.QUIT:
            konets = True

    # Обновляем кадры
    ekran.fill(BELYI)
    for i in range(igra.vysota):
        for j in range(igra.shirina):
            pygame.draw.rect(ekran, SERYI, [igra.x + igra.uvelichenie * j, igra.y + igra.uvelichenie * i, igra.uvelichenie, igra.uvelichenie], 1)
            if igra.pole[i][j] > 0:
                pygame.draw.rect(ekran, cveta[igra.pole[i][j]], [igra.x + igra.uvelichenie * j + 1, igra.y + igra.uvelichenie * i + 1, igra.uvelichenie - 2, igra.uvelichenie - 2])

    if igra.figura:
        for i in range(4):
            for j in range(4):
                p = i * 4 + j
                if p in igra.figura.izobrazhenie():
                    pygame.draw.rect(ekran, cveta[igra.figura.cveta],
                                     [igra.x + igra.uvelichenie * (j + igra.figura.x) + 1,
                                      igra.y + igra.uvelichenie * (i + igra.figura.y) + 1,
                                      igra.uvelichenie - 2, igra.uvelichenie - 2])

    pygame.display.flip()
    chas.tick(fps)

    # Показываем наше видео
    # cv2.imshow("Kulachok", otrazheniye)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    # переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
    # print(results.multi_handedness)

cap.release()
pygame.quit()
cv2.destroyAllWindows()
