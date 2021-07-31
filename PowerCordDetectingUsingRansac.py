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

 # First we will see the initial classification received from the device
with laspy.open('./data/GKR_550_149_region_of_interest_scaled.las') as input_las:
    print('Points from Header:', input_las.header.point_count)
    las = input_las.read()
    print(las)
    print('Points from data:', len(las.points))
    ground_pts = las.classification == 2
    unclassified= las.classification == 1
    water= las.classification == 9
    vegetation= np.logical_and(las.classification == 3,las.classification == 4)

    bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)

    binsUnclassified, countsUnclassified = np.unique(las.return_number[unclassified], return_counts=True)
    
    binsWater, countsWater = np.unique(las.return_number[water], return_counts=True)
    
    binsVeg, countsVeg = np.unique(las.return_number[vegetation], return_counts=True)


    print('Ground Point Return Number distribution:')
    for r,c in zip(bins,counts):
        print('    {}:{}'.format(r,c))
    print('Unclassified Point Return Number distribution:')
    for r,c in zip(binsUnclassified,countsUnclassified):
        print('    {}:{}'.format(r,c))
    print('Water Point Return Number distribution:')
    for r,c in zip(binsWater,countsWater):
        print('    {}:{}'.format(r,c))
    print('Vegetation Point Return Number distribution:')
    for r,c in zip(binsVeg,countsVeg):
        print('    {}:{}'.format(r,c))

# We found out that we mainly  received points classified as  ground.
# In order to find the power cord lines we will need to make a subset on the original image set points. 
# In this refrence it is stated that the return values that we should be searching for are 2 and 3  https://www.usna.edu/Users/oceano/pguth/md_help/html/LidarReturnClassification.htm 
