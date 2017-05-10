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


def draw_lines(img, lines, rgb):
    for line in lines:
        line = line[0]
        draw_line(outer_box, line, rgb)


def merge_related_lines(img, lines):
    for line1 in lines:
        line1 = line1[0]
        if line1[0] == 0 and line1[1] == -100:
            continue
        rho1 = line1[0]
        theta1 = line1[1]
        if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
            pt11x = 0
            pt11y = rho1 / np.sin(theta1)
            pt12x = img.shape[1]
            pt12y = -pt12x / np.tan(theta1) + rho1 / np.sin(theta1)
        else:
            pt11x = rho1 / np.cos(theta1)
            pt11y = 0
            pt12y = img.shape[0]
            pt12x = -pt12y / np.tan(theta1) + rho1 / np.cos(theta1)

        for line2 in lines:
            line2 = line2[0]
            if (line1 == line2).all():
                continue
            rho2 = line2[0]
            theta2 = line2[1]
            if np.fabs(rho2 - rho1) < 20 and np.fabs(theta2 - theta1) < np.pi * 10 / 180:
                if theta2 > np.pi * 45 / 180 and theta2 < np.pi * 135 / 180:
                    pt21x = 0
                    pt21y = rho2 / np.sin(theta2)
                    pt22x = img.shape[1]
                    pt22y = -pt22x / np.tan(theta2) + rho2 / np.sin(theta2)
                else:
                    pt21x = rho2 / np.cos(theta2)
                    pt21y = 0
                    pt22y = img.shape[0]
                    pt22x = -pt22y / np.tan(theta1) + rho2 / np.cos(theta2)

                dst1x = pt21x - pt11x
                dst1y = pt21y - pt11y
                dst2x = pt22x - pt12x
                dst2y = pt22y - pt12y
                if dst1x * dst1x + dst1y * dst1y < 64 * 64 and dst2x * dst2x + dst2y + dst2y < 64 * 64:
                    line1[0] = (line1[0] + line2[0]) / 2
                    line1[1] = (line1[1] + line2[1]) / 2
                    line2[0] = 0
                    line2[1] = -100


path = 'citra/sudoku1.jpg'

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

# hough transform dan gabung line-line yang berdekatan
lines = cv2.HoughLines(outer_box, 1, np.pi / 180, 200)
merge_related_lines(sudoku, lines)
outer_box = cv2.cvtColor(outer_box, cv2.COLOR_GRAY2BGR)
draw_lines(outer_box, lines, (0, 0, 128))

cv2.imshow('sudoku', sudoku)
cv2.imshow('outer', outer_box)
cv2.waitKey()
