import cv2
import numpy as np
import time


def filter(samp, temp, size_th=5, orient_th=3, d_ratio=0.7):

    sift = cv2.SIFT_create()
    kpT = sift.detect(temp, None)
    kpS = sift.detect(samp, None)
    kpS = [k for k in kpS if k.size > size_th]
    kpT = [k for k in kpT if k.size > size_th]

    desT = sift.compute(temp, kpT)[1]
    desS = sift.compute(samp, kpS)[1]
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(desT, desS, k=2)
    good = [m for m,n in matches if m.distance < d_ratio*n.distance]

    filtered = []
    for m in good:
        a1 = kpT[m.queryIdx].angle
        a2 = kpS[m.trainIdx].angle
        diff = abs(a1 - a2)
        diff = min(diff, 360 - diff)

        if diff < orient_th:   filtered.append(m)

    points_temp = np.float32([kpT[m.queryIdx].pt for m in filtered])
    points_samp = np.float32([kpS[m.trainIdx].pt for m in filtered])

    return points_temp, points_samp

def invariance(samp, temp, size_th=5, orient_th=3, d_ratio=0.7):

    sift = cv2.SIFT_create()
    kpT = sift.detect(temp, None)
    kpS = sift.detect(samp, None)
    kpS = [k for k in kpS if k.size > size_th]
    kpT = [k for k in kpT if k.size > size_th]
    for k in kpS: k.angle = 0
    for k in kpT: k.angle = 0

    desT = sift.compute(temp, kpT)[1]
    desS = sift.compute(samp, kpS)[1]
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(desT, desS, k=2)
    filtered = [m for m,n in matches if m.distance < d_ratio*n.distance]

    points_temp = np.float32([kpT[m.queryIdx].pt for m in filtered])
    points_samp = np.float32([kpS[m.trainIdx].pt for m in filtered])

    return points_temp, points_samp