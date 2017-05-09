import cv2
import numpy as np
import math


def draw_line(img, line, rgb):
    if line[1] <> 0:
        m = -1 / np.tan(line[1])
        c = line[0] / np.sin(line[1])
        cv2.line(img, (0, c), (img.shape[1], int(m * img.shape[1] + c)), rgb)
    else:
        cv2.line(img, (line[0], 0), (line[0], img.shape[0]), rgb)


path = 'sudoku.jpg'

# image
sudoku = cv2.imread(path)
sudoku = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)

# preprocessing
sudoku = cv2.GaussianBlur(sudoku, (11, 11), 0)
outer_box = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outer_box = cv2.bitwise_not(outer_box)

# cari kotak terluar
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])
outer_box = cv2.dilate(outer_box, kernel)
count = 0
max_size = -1
h, w = outer_box.shape
mask = np.zeros((h + 2, w + 2), np.uint8)
for i in range(w):
    for j in range(h):
        if outer_box[j][i] > 128:
            area_size = cv2.floodFill(outer_box, mask, (i, j), 64)[0]
            if area_size > max_size:
                max_size = area_size
                max_point = (i, j)
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(outer_box, mask, max_point, 255)
for i in range(w):
    for j in range(h):
        if outer_box[j][i] < 255:
            outer_box[j][i] = 0
outer_box = cv2.erode(outer_box, kernel)

# hough transform
lines = cv2.HoughLines(outer_box, 1, np.pi / 180, 200)
outer_box = cv2.cvtColor(outer_box, cv2.COLOR_GRAY2BGR)
for line in lines:
    draw_line(outer_box, line[0], (0, 0, 128))

cv2.imshow('sudoku', sudoku)
cv2.imshow('outer', outer_box)
cv2.waitKey()
