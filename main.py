from cv2 import cv2
import numpy as np
import imutils


def centroid(contour):
    M = cv2.moments(contour)
    cx = int(round(M['m10'] / M['m00']))
    cy = int(round(M['m01'] / M['m00']))
    centre = (cx, cy)
    return centre


def getScore(scoreboundaries, HoleDist):  # function to assign a score to each hole

    score = 0

    if scoreboundaries[0] > HoleDist:
        score = 10
    for i in range(1, len(scoreboundaries)):
        if scoreboundaries[i - 1] <= HoleDist < scoreboundaries[i]:
            score = len(scoreboundaries) - i
    return score


default = cv2.imread("Tarcza7.jpg")
img = cv2.resize(default, (640, 640))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Edge Cascade
canny = cv2.Canny(v, 50, 175)
cv2.imshow('canny2', canny)

v_mask = cv2.inRange(v, 0, 155)

cv2.imshow('frame', v_mask)
cv2.waitKey(0)

cnts = cv2.findContours(v_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:

    if cv2.contourArea(c) > 10000:
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        area_max = cv2.contourArea(c)

radius_max = np.sqrt(area_max / np.pi)
section_size = radius_max / 10

centre_v_mask = cv2.inRange(canny, 0, 70)
#s = s*1
#s = np.clip(s,0,255)
imghsv = cv2.merge([h,s,v])

cv2.imshow('frame', imghsv)
cv2.waitKey(0)

cnts = cv2.findContours(centre_v_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cv2.drawContours(img, [c], -1, (0, 255, 0), 1)

for c in cnts:
    cv2.drawContours(img, [c], -1, (0, 255, 0), 1)
    print(cv2.contourArea(c))
    if cv2.contourArea(c) > 10:
        centre_coords = centroid(c)

cv2.imshow('frame', img)
cv2.waitKey(0)
h_mask = cv2.inRange(h, 0, 30)
h_mask = cv2.medianBlur(h_mask, 11)
cnts = cv2.findContours(h_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
holes = []
HoleDists = []
scoreboundaries = []

for i in range(1, 10):  # calculate other rings

    cv2.circle(img, centre_coords, int(i * (section_size)), (255, 0, 0), 1)
    scoreboundaries.append(int(i * (section_size)))

for c in cnts:  # plot bullet holes

    if cv2.contourArea(c) > 1:
        x, y, w, h = cv2.boundingRect(c)
        pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]

        centre_holes = centroid(c)
        pts.append(centre_holes)

        pointscore = 0
        for pt in c:
            pt = pt[0]  # contour points have an extra pair of brackets [[x,y]]
            X = pt[0]
            Y = pt[1]

            HoleDist = np.sqrt((X - centre_coords[0]) ** 2 + (Y - centre_coords[1]) ** 2)
            HoleDists.append(HoleDist)
            score = getScore(scoreboundaries, HoleDist)

            if score > pointscore:
                pointscore = score

        cv2.circle(img, (centre_holes), 1, (0, 0, 255), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 1)

        cv2.putText(img, "Score: " + str(pointscore), (centre_holes[0] - 20, centre_holes[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('frame', img)
cv2.waitKey(0)