import cv2
import numpy as np

def convertToXY(shape, line):
    if line[1] <> 0:
        m = -1 / np.tan(line[1])
        c = line[0] / np.sin(line[1])
        pt1x = 0
        pt1y = c
        pt2x = shape[1]
        pt2y = int(m * shape[1] + c)
    else:
        pt1x = line[0]
        pt1y = 0
        pt2x = line[0]
        pt2y = shape[0]
    return (pt1x, pt1y), (pt2x, pt2y)


def drawLine(img, line, rgb):
    pt1, pt2 = convertToXY(img.shape, line)
    cv2.line(img, pt1, pt2, rgb)


def drawLines(img, lines, rgb):
    for line in lines:
        line = line[0]
        drawLine(img, line, rgb)


def mergeRelatedLines(shape, lines):
    temp = lines.copy()
    for line1 in temp:
        line1 = line1[0]
        if line1[0] == 0 and line1[1] == -100:
            continue
        r1 = line1[0]
        theta1 = line1[1]
        if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
            pt11x = 0
            pt11y = r1 / np.sin(theta1)
            pt12x = shape[1]
            pt12y = -pt12x / np.tan(theta1) + r1 / np.sin(theta1)
        else:
            pt11x = r1 / np.cos(theta1)
            pt11y = 0
            pt12y = shape[0]
            pt12x = -pt12y / np.tan(theta1) + r1 / np.cos(theta1)

        for line2 in temp:
            line2 = line2[0]
            if (line1 == line2).all() or (line2[0] == 0 and line2[1] == -100):
                continue
            r2 = line2[0]
            theta2 = line2[1]
            if np.fabs(r2 - r1) < 20 and np.fabs(theta2 - theta1) < np.pi * 10 / 180:
                if theta2 > np.pi * 45 / 180 and theta2 < np.pi * 135 / 180:
                    pt21x = 0
                    pt21y = r2 / np.sin(theta2)
                    pt22x = shape[1]
                    pt22y = -pt22x / np.tan(theta2) + r2 / np.sin(theta2)
                else:
                    pt21x = r2 / np.cos(theta2)
                    pt21y = 0
                    pt22y = shape[0]
                    pt22x = -pt22y / np.tan(theta1) + r2 / np.cos(theta2)

                dst1x = pt21x - pt11x
                dst1y = pt21y - pt11y
                dst2x = pt22x - pt12x
                dst2y = pt22y - pt12y
                if dst1x * dst1x + dst1y * dst1y < 64 * 64 and dst2x * dst2x + dst2y + dst2y < 64 * 64:
                    line1[0] = (line1[0] + line2[0]) / 2
                    line1[1] = (line1[1] + line2[1]) / 2
                    line2[0] = 0
                    line2[1] = -100

    delete_index = []
    for i in range(len(temp)):
        line = temp[i][0]
        if line[0] == 0 and line[1] == -100:
            delete_index.append(i)
    temp = np.delete(temp, delete_index, 0)
    return temp


def findExtremeLines(lines):
    topLine = (1000, 1000)
    bottomLine = (-1000, -1000)
    leftXIntercept = 100000
    rightXIntercept = 0

    for line in lines:
        line = line[0]
        r = line[0]
        theta = line[1]
        xIntercept = r / np.cos(theta)

        if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
            if r < topLine[0]:
                topLine = line
            if r > bottomLine[0]:
                bottomLine = line
        elif theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
            if xIntercept > rightXIntercept:
                rightLine = line
                rightXIntercept = xIntercept
            elif xIntercept <= leftXIntercept:
                leftLine = line
                leftXIntercept = xIntercept

    return topLine, bottomLine, leftLine, rightLine


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def getLineIntersect(shape, line1, line2):
    line1 = convertToXY(shape, line1)
    line2 = convertToXY(shape, line2)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
