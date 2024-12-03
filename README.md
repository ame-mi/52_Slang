# Tetris_MediaPipe
A program that will help you to play tetris without touching your computer.

import pygame
import random
import cv2
import mediapipe as mp
import numpy as np

# Для точечек
def poluchit_tochki(landmark, forma):
    tochki = []
    for mark in landmark:
        tochki.append([mark.x * forma[1], mark.y * forma[0]])
    return np.array(tochki, dtype=np.int32)

# Для размера
def razmer_ladoni(landmark, forma):
    x1, y1 = landmark[0].x * forma[1], landmark[0].y * forma[0]
    x2, y2 = landmark[5].x * forma[1], landmark[5].y * forma[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

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
        self.vysota = vysota  # Высота игрового поля
        self.shirina = shirina  # Ширина игрового поля
        self.x = 100  # Смещение игрового поля по x
        self.y = 60  # Смещение игрового поля по y
        self.uvelichenie = 20  # Размер клетки
        self.figura = None  # Текущая фигура
        for i in range(vysota):
            self.pole.append([0] * shirina)

    # Создаем новую фигру
    def novaya_figura(self):
        self.figura = Figura(3, 0)

    # Проверяем, что они не пересекаются
    def peresechenie(self):
        peresechenie = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figura.izobrazhenie():
                    if (i + self.figura.y >= self.vysota or
                        j + self.figura.x >= self.shirina or
                        j + self.figura.x < 0 or
                        self.pole[i + self.figura.y][j + self.figura.x] > 0):
                        peresechenie = True
        return peresechenie

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
        self.novaya_figura()
        if self.peresechenie():
            self.sostoyanie = "gameover"

    # Двигаем в бок с помощью кулачков
    def v_bok(self, dx):
        star_x = self.figura.x
        self.figura.x += dx
        if self.peresechenie():
            self.figura.x = star_x

# Инициализация Mediapipe и OpenCV
mp_ruki = mp.solutions.hands
ruki = mp_ruki.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Инициализация Pygame
pygame.init()

# Определение цветов
CHERNYY = (0, 0, 0)
BELYI = (255, 255, 255)
SERYI = (128, 128, 128)

# Размеры экрана
razmer = (400, 500)
ekran = pygame.display.set_mode(razmer)
pygame.display.set_caption("Тетрисчк")

# Моя камера ура
kamera = cv2.VideoCapture(0)

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
while not konets:
    ret, kad = kamera.read()
    if not ret:
        break
    otrazheniye = cv2.flip(kad, 1) # отражаем, чтобы не путать право и лево
    vysota, shirina, _ = otrazheniye.shape
    kad_rgb = cv2.cvtColor(otrazheniye, cv2.COLOR_BGR2RGB)
    result = ruki.process(kad_rgb)

    # Распознаем кулачки. Вдохновение взято из кода "Детектирование кулака", переделано мной под нашу ситуацию
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            rukapravaya_levaya = hand_handedness.classification[0].label
            tochki = poluchit_tochki(hand_landmarks.landmark, otrazheniye.shape)
            radius = razmer_ladoni(hand_landmarks.landmark, otrazheniye.shape)
            (x, y), rad = cv2.minEnclosingCircle(tochki)
            zazhat = 2 * rad / radius < 1.3

            if rukapravaya_levaya == "Right":
                if zazhat and not pred_fist_pravaya:
                    igra.v_bok(1)
                    pred_fist_pravaya = True
                elif not zazhat:
                    pred_fist_pravaya = False
            elif rukapravaya_levaya == "Left":
                if zazhat and not pred_fist_levaya:
                    igra.v_bok(-1)
                    pred_fist_levaya = True
                elif not zazhat:
                    pred_fist_levaya = False

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
    cv2.imshow("Kulachok", otrazheniye)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
pygame.quit()
cv2.destroyAllWindows()

