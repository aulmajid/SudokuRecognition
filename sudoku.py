import cv2
import numpy as np
import utils

srcPath = 'citra/sudoku.jpg'
dstPath = 'sudoku.jpg'


# image
original = cv2.imread(srcPath)
shape = original.shape
height = shape[0]
width = shape[1]
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]])

# cari kotak terluar
sudokuRect = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
sudokuRect = cv2.GaussianBlur(sudokuRect, (11, 11), 0)
sudokuRect = cv2.adaptiveThreshold(sudokuRect, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
sudokuRect = cv2.bitwise_not(sudokuRect)
sudokuRect = cv2.dilate(sudokuRect, kernel)
max_size = -1
mask = np.zeros((height + 2, width + 2), np.uint8)
for i in range(width):
    for j in range(height):
        if sudokuRect[j][i] > 128:
            area_size = cv2.floodFill(sudokuRect, mask, (i, j), 64)[0]
            if area_size > max_size:
                max_size = area_size
                max_point = (i, j)
mask = np.zeros((height + 2, width + 2), np.uint8)
cv2.floodFill(sudokuRect, mask, max_point, 255)
for i in range(width):
    for j in range(height):
        if sudokuRect[j][i] < 255:
            sudokuRect[j][i] = 0

# cari garis2 luar dengan hough transform
lines = cv2.HoughLines(sudokuRect, 1, np.pi / 180, 200)
lines = utils.mergeRelatedLines(shape, lines)
top, bottom, left, right = utils.findExtremeLines(lines)
extremes = [[top], [bottom], [left], [right]]

# cari titik2 terluar
topLeft = utils.getLineIntersect(shape, top, left)
topRight = utils.getLineIntersect(shape, top, right)
bottomLeft = utils.getLineIntersect(shape, bottom, left)
bottomRight = utils.getLineIntersect(shape, bottom, right)

# warping
src = np.array([
    topLeft,
    topRight,
    bottomLeft,
    bottomRight
], "float32")
newSize = 400
dst = np.array([
    [0, 0],
    [newSize, 0],
    [0, newSize],
    [newSize, newSize]
], "float32")
M = cv2.getPerspectiveTransform(src, dst)
sudoku = cv2.warpPerspective(original, M, (newSize, newSize))

# print
cv2.imshow('original', original)
cv2.imshow('sudoku', sudoku)
cv2.imwrite(dstPath, sudoku)
cv2.waitKey()
