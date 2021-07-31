#  Assignment: Classifying power cords using Ransac algorithm on images gathered with Lidar Tools
#  Course: Remote Sensing
#  Insititution: Facuty of Electric Engineering and Computer Science, University of Maribor
#  Author: Aleksandar Tendjer
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import laspy
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from skimage.measure import LineModelND, ransac


# First we will see the initial classification received from the device
with laspy.open('./data/GKR_550_149_region_of_interest_scaled.las') as inputLas:
    print('Points from Header:', inputLas.header.point_count)
    las = inputLas.read()
    print(las)
    print('Points from data:', len(las.points))
    groundPts = las.classification == 2
    unclassified = las.classification == 1
    water = las.classification == 9
    vegetation = np.logical_and(
        las.classification == 3, las.classification == 4)

    bins, counts = np.unique(las.return_number[groundPts], return_counts=True)

    binsUnclassified, countsUnclassified = np.unique(
        las.return_number[unclassified], return_counts=True)

    binsWater, countsWater = np.unique(
        las.return_number[water], return_counts=True)

    binsVeg, countsVeg = np.unique(
        las.return_number[vegetation], return_counts=True)

    print('Ground Point Return Number distribution:')
    for r, c in zip(bins, counts):
        print('    {}:{}'.format(r, c))
    print('Unclassified Point Return Number distribution:')
    for r, c in zip(binsUnclassified, countsUnclassified):
        print('    {}:{}'.format(r, c))
    print('Water Point Return Number distribution:')
    for r, c in zip(binsWater, countsWater):
        print('    {}:{}'.format(r, c))
    print('Vegetation Point Return Number distribution:')
    for r, c in zip(binsVeg, countsVeg):
        print('    {}:{}'.format(r, c))
    # We found out that we mainly  received points classified as  ground.
    # In order to find the power cord lines we will need to make a subset on the original image set points.
    # In this refrence it is stated that the return values that we should be searching for are 2 and 3  https://www.usna.edu/Users/oceano/pguth/md_help/html/LidarReturnClassification.htm
    pointsOfInterest = las[las.return_number[groundPts] > 1]
    averageGroundHeight = las[las.return_number[groundPts] == 1]['Z'].mean()
    pointsForRansac = []
    for p in pointsOfInterest:
        if p['Z'] - averageGroundHeight > 100:
            pointsForRansac.append([p['X'],
                                   p['Y'], p['Z']])
    possibleCordPoints = np.array(pointsForRansac)
    # sorting by z value
    # possibleCordPoints = possibleCordPoints[possibleCordPoints[:, 2].argsort(
    # )]
    startRange = 0
    stepIncrease = 2000

    endRange = stepIncrease
    while endRange <= len(possibleCordPoints):
        model_robust, inliers = ransac(
            possibleCordPoints[startRange:endRange], LineModelND, min_samples=2, residual_threshold=1, max_trials=1500)
        trueInliners = np.where(
            inliers[True] == True)
        for idx in trueInliners[1]:
            powerCordPoint = possibleCordPoints[idx]
            xIndexOfOriginalImg = np.where(
                las['X'] == possibleCordPoints[idx][0])
            yIndexOfOriginalImg = np.where(
                las['Y'] == possibleCordPoints[idx][1])
            zIndexOfOriginalImg = np.where(
                las['Z'] == possibleCordPoints[idx][2])
            # intersecting indexes to find the actual point in the original image
            lasIdx = np.intersect1d(np.intersect1d(np.where(las['Z'] == possibleCordPoints[idx][2]), np.where(
                las['X'] == possibleCordPoints[idx][0])), np.where(las['Y'] == possibleCordPoints[idx][1]))
            # setting its classification value as power line
            las[lasIdx].classification = 16
        startRange += stepIncrease
        endRange += stepIncrease
        if endRange >= len(possibleCordPoints) and startRange < len(possibleCordPoints):
            endRange = startRange+(len(possibleCordPoints)-startRange-1)
    outputFile = laspy.LasData(inputLas.header)
    outputFile.points = las
    outputFile.write("powerLineClassified.las")
