import cv2
import time
import mediapipe


def json_writer(pre_list):
    with open('coordinates.data', 'w') as f:
        global length
        for i in range(0, length):
            for j in range(0, 4):
                f.write(str(pre_list[i][j]) + '\t')
            f.write('\n')
    f.close()


def json_reader():
    readlist = []
    with open('coordinates.data', 'r') as f:
        for i in range(0, length):
            content = f.readline().split('\t')
            content.pop(-1)
            readlist.append(content)
    return readlist


if __name__ == '__main__':
    length = 3
    json_writer([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    print(json_reader())