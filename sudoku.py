import cv2
import numpy as np
import utils

path = 'citra/sudoku.jpg'

# image
sudoku = cv2.imread(path)

# cari kotak terluar
outer_box = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
outer_box = cv2.GaussianBlur(outer_box, (11, 11), 0)
outer_box = cv2.adaptiveThreshold(outer_box, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outer_box = cv2.bitwise_not(outer_box)
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

# cari garis2 dengan hough transform
lines = cv2.HoughLines(outer_box, 1, np.pi / 180, 200)
lines = utils.mergeRelatedLines(sudoku, lines)

# cari garis2 terluar
top, bottom, left, right = utils.findExtremeLines(lines)
extremes = [[top], [bottom], [left], [right]]

# warping
src = np.zeros((4, 2), "float32")
src[0] = utils.getLineIntersect(sudoku, top, left)
src[1] = utils.getLineIntersect(sudoku, top, right)
src[2] = utils.getLineIntersect(sudoku, bottom, right)
src[3] = utils.getLineIntersect(sudoku, bottom, left)
newSize = 400
dst = np.array([
    [0, 0],
    [newSize, 0],
    [newSize, newSize],
    [0, newSize]
], "float32")
M = cv2.getPerspectiveTransform(src, dst)
sudoku = cv2.warpPerspective(sudoku, M, (newSize, newSize))

cv2.imshow('sudoku', sudoku)
cv2.waitKey()
