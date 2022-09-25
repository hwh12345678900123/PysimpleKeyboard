import cv2
import time
import mediapipe


def json_writer(pre_list):
    with open('data.json', 'w') as f:
        global length
        for i in range(0, length - 1):
            for j in range(0, 3):
                f.write(pre_list[i][j] + '\t')
            f.write('\n')
    f.close()


def json_reader():
    readlist = []
    with open('data.json', 'r') as f:
        for i in range(0, length - 1):
            readlist.append(f.readline().split('\t'))
    return readlist


if __name__ == '__main__':
    length = 3
    json_writer([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
