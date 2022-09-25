import random
import time
# from base import *
# from MMEdu import MMDetection as det, MMClassification as cls
import cv2
import mediapipe as mp
import os
import playsound
import math
from pykeyboard import PyKeyboard
# import numpy as np
import threading

k = PyKeyboard()
# firstlist = [[[40, 320], [95, 320], [25, 365], [80, 365]], [[105, 325], [160, 325], [170, 360], [220, 360]],
#              [[170, 320], [225, 320], [170, 360], [220, 360]], [[240, 320], [295, 320], [235, 360], [290, 360]],
#              [[60, 280], [105, 280], [45, 315], [95, 315]],
#              [[120, 275], [170, 275], [110, 310], [160, 310]],
#              [[185, 280], [235, 280], [180, 320], [235, 320]], [[240, 275], [295, 275], [240, 310], [295, 310]],
#              [[70, 235], [115, 235], [60, 270], [105, 270]]]
# firstlist = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []],
#              [[], [], [], []], [[], [], [], []], [[], [], [], []]]
# firstlist = [[[472, 430], [527, 430], [461, 386], [513, 387]], [[404, 428], [458, 426], [395, 383], [448, 386]],
#              [[330, 428], [390, 427], [327, 384], [380, 384]], [[258, 427], [315, 427], [256, 382], [313, 381]],
#              [[458, 377], [509, 377], [448, 337], [497, 339]], [[393, 374], [446, 375], [386, 337], [436, 337]],
#              [[325, 374], [380, 373], [322, 337], [373, 337]], [[257, 372], [312, 372], [256, 333], [309, 333]],
# #              [[446, 330], [495, 330], [437, 294], [483, 295]]]
# firstlist = [[[471, 430], [525, 429], [460, 386], [512, 383]], [[403, 425], [458, 430], [393, 386], [446, 387]],
#              [[330, 428], [387, 426], [326, 383], [379, 384]], [[256, 426], [315, 428], [254, 383], [311, 384]],
#              [[458, 377], [509, 377], [446, 337], [496, 340]], [[391, 375], [443, 376], [385, 335], [434, 338]],
#              [[323, 374], [377, 374], [320, 335], [372, 330]], [[255, 373], [312, 374], [256, 333], [307, 336]],
#              [[444, 330], [494, 330], [435, 295], [483, 296]]]
firstlist = [[[120, 430], [175, 430], [135, 385], [185, 385]], [[190, 425], [240, 425], [120, 385], [250, 385]],
             [[260, 425], [315, 425], [265, 385], [320, 385]], [[330, 425], [390, 425], [330, 380], [390, 380]],
             [[140, 375], [190, 375], [150, 335], [200, 335]], [[200, 370], [255, 470], [210, 330], [260, 330]],
             [[265, 370], [320, 370], [270, 330], [325, 335]], [[335, 370], [390, 370], [340, 330], [390, 330]],
             [[150, 325], [200, 325], [160, 290], [210, 290]]]
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def in_list(list1, list2, start=0, end=-1):
    head_element = list1[0]
    try:
        part = list2.index(head_element, start, end)
    except ValueError:
        return False
    for i in range(len(list1)):
        try:
            if not list2[i + part] == list1[i]:
                if i + part == len(list2) - 1:
                    return False
                try:
                    return in_list(list1, list2, start=part + 1, end=end)
                except ValueError:
                    return False
        except IndexError:
            return False
    return True


# def initialize(image):
#     # image = cv2_get()
#     model = det(backbone='FasterRCNN')
#     checkpoint = '../checkpoints/det_model/myProject/latest.pth'
#     class_path = '../dataset/det/myProject/classes.txt'
#     result = model.inference(image=image, show=True, class_path=class_path, checkpoint=checkpoint)
#     print(result)
#     model.print_result()
#
#
# def retrain_det():
#     model = det(backbone='Mask_RCNN')
#     model.load_dataset(path='../dataset/det/myProject')
#     model.num_classes = 9
#     model.save_fold = '../det_points/myProject'
#     # checkpoint = '../checkpoints/det_model/myProject/latest.pth'
#     model.train(epochs=250, lr=0.05, validate=True)  # , checkpoint=checkpoint)
#
#
# # retrain_det()
#
# img = "/home/user/Desktop/Project/mmedu/Project/dataset/det/myProject/images/test/81.jpg"
# initialize(img)

class Point(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


def distance(point1: Point, point2: Point):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def check2(a1: Point, a2: Point):
    x_depth = abs(a1.x - a2.x)
    y_depth = abs(a1.y - a2.y)
    if abs(x_depth - y_depth) < 10:
        return "same"
    elif x_depth < y_depth:
        return "y"
    else:
        return "x"


class Vector(object):
    def __init__(self, a1: Point, a2: Point, a3: Point, a4: Point, a5: Point):
        self.a1, self.a2, self.a3, self.a4, self.a5 = a1, a2, a3, a4, a5
        self.between = []

    def new(self, new_point: Point):
        self.a1, self.a2, self.a3, self.a4 = self.a2, self.a3, self.a4, self.a5
        self.a5 = new_point

    def iskeydown(self):
        print(self.between)
        if in_list(['y', 'x', 'y'], self.between) or (
                self.between[0] == self.between[-1] == "y" and 'x' in self.between):
            print("loggggg")
            return True
        else:
            return False

    def check(self):
        list3 = [self.a1, self.a2, self.a3, self.a4, self.a5]
        list2 = []
        for j in range(0, 4):
            list2.append(check2(list3[j], list3[j + 1]))
        self.between = list2
        # print(self.between)


vector = [Vector(Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)),
          Vector(Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)),
          Vector(Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)),
          Vector(Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)),
          Vector(Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0))]


class Key(object):
    def __init__(self, name, a1: Point, a2: Point, a3: Point, a4: Point):
        self.name, self.a1, self.a2, self.b1, self.b2 = name, a1, a2, a3, a4
        self.locktime = 0

    def inside_keyed(self, x, y):
        print(x, y, self.name)
        if x > self.a1.x and self.a1.y > y \
                and self.a2.x > x and self.a2.y > y \
                and self.b1.x < x and self.b1.y < y \
                and self.b2.x > x and self.b2.y < y:
            print("goooooooooooooooooool")
            return True
        return False

    def to_string(self):
        return self.name

    def lock(self):
        self.locktime = 5

    def unlock(self):
        self.locktime -= 1
        if self.locktime < 0:
            self.locktime = 0


a1_1, a2_1, b1_1, b2_1 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_2, a2_2, b1_2, b2_2 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_3, a2_3, b1_3, b2_3 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_4, a2_4, b1_4, b2_4 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_5, a2_5, b1_5, b2_5 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_6, a2_6, b1_6, b2_6 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_7, a2_7, b1_7, b2_7 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_8, a2_8, b1_8, b2_8 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
a1_9, a2_9, b1_9, b2_9 = Point(0, 0), Point(0, 0), Point(0, 0), Point(0, 0)
one = Key(1, a1_1, a2_1, b1_1, b2_1)
two = Key(2, a1_2, a2_2, b1_2, b2_2)
three = Key(3, a1_3, a2_3, b1_3, b2_3)
four = Key(4, a1_4, a2_4, b1_4, b2_4)
five = Key(5, a1_5, a2_5, b1_5, b2_5)
six = Key(6, a1_6, a2_6, b1_6, b2_6)
seven = Key(7, a1_7, a2_7, b1_7, b2_7)
eight = Key(8, a1_8, a2_8, b1_8, b2_8)
nine = Key(9, a1_9, a2_9, b1_9, b2_9)

initialized_keys = []


def initialize(firstlist):
    global a1_1, a2_1, b1_1, b2_1, a1_2, a2_2, b1_2, b2_2, a1_3, a2_3, b1_3, b2_3, a1_4, a2_4, b1_4, b2_4, a1_5, a2_5, b1_5, b2_5, a1_6, a2_6, b1_6, b2_6, a1_7, a2_7, b1_7, b2_7, a1_8, a2_8, b1_8, b2_8, a1_9, a2_9, b1_9, b2_9
    global one, two, three, four, five, six, seven, eight, nine
    global initialized_keys
    a1_1 = Point(firstlist[0][0][0], firstlist[0][0][1])
    a2_1 = Point(firstlist[0][1][0], firstlist[0][1][1])
    b1_1 = Point(firstlist[0][2][0], firstlist[0][2][1])
    b2_1 = Point(firstlist[0][3][0], firstlist[0][3][1])
    one = Key(1, a1_1, a2_1, b1_1, b2_1)
    a1_2 = Point(firstlist[1][0][0], firstlist[1][0][1])
    a2_2 = Point(firstlist[1][1][0], firstlist[1][1][1])
    b1_2 = Point(firstlist[1][2][0], firstlist[1][2][1])
    b2_2 = Point(firstlist[1][3][0], firstlist[1][3][1])
    two = Key(2, a1_2, a2_2, b1_2, b2_2)
    a1_3 = Point(firstlist[2][0][0], firstlist[2][0][1])
    a2_3 = Point(firstlist[2][1][0], firstlist[2][1][1])
    b1_3 = Point(firstlist[2][2][0], firstlist[2][2][1])
    b2_3 = Point(firstlist[2][0][0], firstlist[2][0][1])
    three = Key(3, a1_3, a2_3, b1_3, b2_3)
    a1_4 = Point(firstlist[3][0][0], firstlist[3][0][1])
    a2_4 = Point(firstlist[3][1][0], firstlist[3][1][1])
    b1_4 = Point(firstlist[3][2][0], firstlist[3][2][1])
    b2_4 = Point(firstlist[3][3][0], firstlist[3][3][1])
    four = Key(4, a1_4, a2_4, b1_4, b2_4)
    a1_5 = Point(firstlist[4][0][0], firstlist[4][0][1])
    a2_5 = Point(firstlist[4][1][0], firstlist[4][1][1])
    b1_5 = Point(firstlist[4][2][0], firstlist[4][2][1])
    b2_5 = Point(firstlist[4][3][0], firstlist[4][3][1])
    five = Key(5, a1_5, a2_5, b1_5, b2_5)
    a1_6 = Point(firstlist[5][0][0], firstlist[5][0][1])
    a2_6 = Point(firstlist[5][1][0], firstlist[5][1][1])
    b1_6 = Point(firstlist[5][2][0], firstlist[5][2][1])
    b2_6 = Point(firstlist[5][3][0], firstlist[5][3][1])
    six = Key(6, a1_6, a2_6, b1_6, b2_6)
    a1_7 = Point(firstlist[6][0][0], firstlist[6][0][1])
    a2_7 = Point(firstlist[6][1][0], firstlist[6][1][1])
    b1_7 = Point(firstlist[6][2][0], firstlist[6][2][1])
    b2_7 = Point(firstlist[6][3][0], firstlist[6][3][1])
    seven = Key(7, a1_7, a2_7, b1_7, b2_7)
    a1_8 = Point(firstlist[7][0][0], firstlist[7][0][1])
    a2_8 = Point(firstlist[7][1][0], firstlist[7][1][1])
    b1_8 = Point(firstlist[7][2][0], firstlist[7][2][1])
    b2_8 = Point(firstlist[7][3][0], firstlist[7][3][1])
    eight = Key(8, a1_8, a2_8, b1_8, b2_8)
    a1_9 = Point(firstlist[8][0][0], firstlist[8][0][1])
    a2_9 = Point(firstlist[8][1][0], firstlist[8][1][1])
    b1_9 = Point(firstlist[8][2][0], firstlist[8][2][1])
    b2_9 = Point(firstlist[8][3][0], firstlist[8][3][1])
    nine = Key(9, a1_9, a2_9, b1_9, b2_9)

    initialized_keys = [one, two, three, four, five, six, seven, eight, nine]
    global flag
    flag = 2


count = 0


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global count
    print(count, firstlist)
    if count >= 36:
        print("log")
        global flag
        if flag == 0:
            cv2.destroyAllWindows()
            flag = 1
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        ret, img = cap.read()
        if not ret:
            raise Exception("Can not get the image from the camera")
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        count += 1
        print((int(count - count % 4) / 4))
        firstlist[int(((count - 1) / 4))][count % 4 - 1] = [x, y]


def confirmed():
    for i in range(0, 5):
        new_point = Point(x=now[i][0], y=now[i][1])
        vector[i].new(new_point)
        vector[i].check()
        if vector[i].iskeydown():
            return i
    return False


def inside_keyed():
    for i in initialized_keys:
        for k in range(0, 5):
            if i.inside_keyed(now[k][0], now[k][1]):
                print("almost donnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnne")
                return i
    return False


def press_key(key):
    if key.locktime > 0:
        # yinxiao
        playsound.playsound("../media/lock.wav")
        return
    k.tap_key(str(key.name))
    playsound.playsound("media/keypress/" + random.choice(os.listdir('../media/keypress')))
    key.lock()


def preinitialize():
    print("请依次点击1至9按键您视野中的的左上，右上，左下，右下点")
    time.sleep(1)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    ret, img = cap.read()
    cv2.imshow("image", img)
    k = cv2.waitKey(0)


flag = 0


def pre_initialize_to_initialize():
    global cap, flag
    if flag == 1:
        cv2.destroyAllWindows()
        cap.release()
        cap = cv2.VideoCapture(-1)
        initialize(firstlist=firstlist)
        flag = 2
    if flag == 2:
        return


def main():
    global flag, cap, hand
    if flag == 2:
        while True:
            time.sleep(0.2)
            success, image = cap.read()
            if not success:
                raise Exception("unable to open the camera")
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            global previous, now
            # print(result.multi_hand_landmarks)
            previous = now
            now = []
            # analysise
            if previous and now:
                # print(now)
                pass
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)
                    # print("||||||||||||||||")
                    for i in range(4, 24, 4):
                        print(hand_landmarks.landmark[i], i)
                        # print("log")
                        # print(hand_landmarks.landmark[i].x * image.shape[0])
                        # print(image.shape[0])
                        now.append(
                            # [(1 - hand_landmarks.landmark[i].x) * image.shape[1],
                            [(1 - hand_landmarks.landmark[i].x) * image.shape[1],
                             (hand_landmarks.landmark[i].y) * image.shape[0],
                             hand_landmarks.landmark[i].z])
                    # print(now)
                    cv2.imwrite('result.png', cv2.flip(image, 1))
                ################################################################
                # confirm
                for i in initialized_keys:
                    i.unlock()
                if now and previous:
                    if confirmed():
                        key = inside_keyed()
                        if key:
                            press_key(key)
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            # print(result[0])
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        flag = 3


if __name__ == '__main__':
    cap = cv2.VideoCapture(-1)
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    previous = []
    now = []
    # t1 = threading.Thread(target=preinitialize)
    # t1.start()
    # t1.join()
    initialize(firstlist)
    main()
