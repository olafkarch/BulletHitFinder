import cv2 as cv
import numpy as np
import math
import sys
from PIL import Image
import matplotlib.pyplot as plt

MAX_POINTS = 10

def main(argv):

    default_file = 'tarcza1-3.png'
    default_size = 600, 600


    im = Image.open(default_file)
    im = im.resize(default_size, Image.ANTIALIAS)
    im.save('600' + default_file)

    filename = argv[0] if len(argv) > 0 else '600' + default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    # skala szarości
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)

    # Bilateral
    bilateral = cv.bilateralFilter(gray, 7, 15, 10)
    cv.imshow('bilateral', bilateral)

    blank = np.zeros(bilateral.shape[:2], dtype='uint8')
    cv.imshow('blank', blank)

    # mask = cv.circle(blank, (bilateral.shape[1] // 2, bilateral.shape[0] // 2), 320, 255, -1)
    # cv.imshow('Mask', mask)
    #
    # masked = cv.bitwise_and(bilateral, bilateral, mask=mask)
    # cv.imshow('masked', masked)

    # Edge Cascade
    canny = cv.Canny(bilateral, 50, 175)
    cv.imshow('canny1', canny)

    # ret, tresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    # cv.imshow('tresch', tresh)

    contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f'{len(contours)} contour(s) found')

    # cv.drawContours(blank, contours, -1, (255,0,0), 1)
    # cv.imshow('contours drawn', blank)

    rows = canny.shape[0]

 # tarcza strzelecka

    circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 0.01,
                              param1=100, param2=50,
                              minRadius=7, maxRadius=300)

    # print(f'{circles}"')

    biggestCircle = findBiggestCircle(circles)
    # print(f'{biggestCircle} biggest circle')

    mask = cv.circle(blank, (math.floor(biggestCircle[0]), math.floor(biggestCircle[1])), math.floor(biggestCircle[2]), 255, -1)
    cv.imshow('rysowanie granicy', mask)

    masked = cv.bitwise_and(bilateral, bilateral, mask=mask)
    cv.imshow('granice', masked)

    # Edge Cascade
    canny = cv.Canny(masked, 50, 175)
    cv.imshow('canny2', canny)

    if biggestCircle is not None:
        circles = np.uint16(np.around(circles))

        # print(f'{biggestCircle} biggest circle')

        delta_r = biggestCircle[2] / 10
        biggest_circle_center = [biggestCircle[0], biggestCircle[1]]

        center = (math.floor(biggestCircle[0]), math.floor(biggestCircle[1]))
        # print(f'{center} center')
        # circle center
        cv.circle(src, center, 1, (255, 0, 0), 3)
        # circle outline
        radius = math.floor(biggestCircle[2])
        cv.circle(src, center, radius, (0, 0, 255), 3)



# dziury po kulach

    hits = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1, 10,
                           param1=300, param2=10,
                           minRadius=10, maxRadius=15)
    # print(f'{hits}"')

    score = countHitScore(hits.tolist(), delta_r, biggest_circle_center)

    print(f'The score is: {score}"')

    if hits is not None:
        hits = np.uint16(np.around(hits))
        for i in hits[0, :]:
            # print(f'promien trafienia {i[2]}"')
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", src)
    cv.waitKey(0)

    return 0

def findBiggestCircle(circles):

    # print(f'{circles}')
    listOfCircles = circles[0]
    biggestCircle = listOfCircles[0]

    for circle in listOfCircles:
        # print(f'{circle} circle')
        # print(f'2 {circle}')
        # print(f'3 {biggestCircle}')
        if circle[2] > biggestCircle[2]:
            # print('4')
            biggestCircle = circle
    print(biggestCircle)
    return biggestCircle.tolist()

def countHitScore(hits, delta_r, target_center):
    score = 0
    print(f'{hits} hits')


    for hit in hits[0]:
        # print(f'{hit} hit')
        # print(f'{(target_center)} center')
        x_dist = hit[0] - target_center[0] if hit[0] > target_center[0] else target_center[0] - hit[0]
        y_dist = hit[1] - target_center[1] if hit[1] > target_center[1] else target_center[1] - hit[1]

        total_dist = math.hypot(x_dist, y_dist) - hit[2]

        punkty = math.ceil(total_dist / delta_r)

        if punkty < 1:
            punkty = 1

        score += 11 - punkty

        # print(f'{total_dist / delta_r} math')
        # print(f'{total_dist / delta_r} total_dist / delta_r')
        print(f'{11 - punkty} zdobyte punkty')
        # print(f'{x_dist} x {y_dist} y')

    return score


if __name__ == "__main__":
    main(sys.argv[1:])
