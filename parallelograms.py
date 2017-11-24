#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
from PIL import Image

np.set_printoptions(threshold = np.nan)

###########################################################################
# 1. OPEN AND SIZE FILE
###########################################################################
def openAndSizeFile(file):
    print 'open and size file'

    #print file[-4:]

    f = Image.open(file)  # open the .jpg
    width, height = f.size  # get the width and height
    #print height
    #print width
    f.close()

    # convert values to rows, cols
    rows = int(height)
    originalRows = int(rows)
    cols = int(width)
    originalCols = int(cols)
    #print str(rows) + 'x' + str(cols)

    file = file[:-4]+".raw"

    pic = open(file, 'rb')  # open file
    f = np.fromfile(pic, dtype=np.uint8)  # read image
    pic.close()
    #print f
    #print len(f)

    img = f.reshape((rows,cols*3))  # reshape into a 2D array
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img, rows, cols


###########################################################################
# 2. CONVERT TO GRAY
###########################################################################
def convertToGray(img, rows, cols):
    print 'convert to gray'

    # create numpy array
    gray = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):
            list1 = []
            for rgb in range(3):
                i2 = col*3+rgb
                list1.append(img[row][i2])
            # assign new intensity value between 0-1
            gray[row][col] = (list1[0]*0.30 + list1[1]*0.59 + list1[2]*0.11)/255

    #del f
    #del img

    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return gray


###########################################################################
# 3. RESIZE FILE
###########################################################################
def resizeFile(gray, rows, cols):
    print 'resizing file'

    # resize it
    # NOTE: if row or col size was odd, it will cut off the last row or col
    while rows > 200 and cols > 200:
        # half size
        rows //= 2  #+ rows % 2
        cols //= 2  #+ cols % 2
        # make new matrix
        smallArr = np.zeros((rows, cols))
        print str(gray.shape) + ' --> ' + str(smallArr.shape)
        #print smallArr

        # find mean of 4 pixels and resize to half (round up)
        for row in range(rows):
            for col in range(cols):
                oldRow = row*2
                oldCol = col*2
                smallArr[row][col] = gray[oldRow][oldCol]

        # copy for next loop
        gray = smallArr[:]

    #del gray
    #cv2.imshow("smallArr", smallArr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return smallArr, rows, cols


###########################################################################
# 4. MEAN FILTER
###########################################################################
def meanFilter(smallArr, rows, cols):
    print 'mean filter'

    meanArr = np.zeros((rows, cols))

    # apply a 5x5 mean filter
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            # NOTE: divide by 25 at the end
            meanArr[row][col] = \
                (smallArr[row - 2][col - 2] + smallArr[row - 2][col - 1] + smallArr[row - 2][col] + smallArr[row - 2][col + 1] + smallArr[row - 2][col + 2] +
                 smallArr[row - 1][col - 2] + smallArr[row - 1][col - 1] + smallArr[row - 1][col] + smallArr[row - 1][col + 1] + smallArr[row - 1][col + 2] +
                 smallArr[row + 0][col - 2] + smallArr[row + 0][col - 1] + smallArr[row + 0][col] + smallArr[row + 0][col + 1] + smallArr[row + 0][col + 2] +
                 smallArr[row + 1][col - 2] + smallArr[row + 1][col - 1] + smallArr[row + 1][col] + smallArr[row + 1][col + 1] + smallArr[row + 1][col + 2] +
                 smallArr[row + 2][col - 2] + smallArr[row + 2][col - 1] + smallArr[row + 2][col] + smallArr[row + 2][col + 1] + smallArr[row + 2][col + 2] ) / 25
    #del smallArr

    # display new image after filter
    #cv2.imshow("meanArr", meanArr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return meanArr


###########################################################################
# 5. GAUSSIAN FILTER
###########################################################################
def gaussianFilter(meanArr, rows, cols):
    print 'gaussian filter'

    # can try regular gaussian, laplacian, and LoG

    # gaussian values:
    # 1 2 1
    # 2 4 2
    # 1 2 1
    gauss = np.zeros((rows, cols))
    #gauss2 = np.zeros((rows, cols))

    # Gaussian
    for row in range(1, rows-1):
        for col in range(1, cols-1):
            gauss[row][col] = ( meanArr[row - 1][col - 1] *  1 + meanArr[row - 1][col] * 2 + meanArr[row - 1][col + 1] * 1 +
                                meanArr[row + 0][col - 1] *  2 + meanArr[row + 0][col] * 4 + meanArr[row + 0][col + 1] * 2 +
                                meanArr[row + 1][col - 1] *  1 + meanArr[row + 1][col] * 2 + meanArr[row + 1][col + 1] * 1 ) /16 # NOTE: divide to normalize

    # # Laplacian
    # for row in range(1, rows-1):
    #     for col in range(1, cols-1):
    #         gauss[row][col] = ( meanArr[row - 1][col - 1] *  0 + meanArr[row - 1][col] * 1 + meanArr[row - 1][col + 1] * 0 +
    #                             meanArr[row + 0][col - 1] *  1 + meanArr[row + 0][col] *-4 + meanArr[row + 0][col + 1] * 1 +
    #                             meanArr[row + 1][col - 1] *  0 + meanArr[row + 1][col] * 1 + meanArr[row + 1][col + 1] * 0 )

    # # LoG
    # for row in range(2, rows-2):
    #     for col in range(2, cols-2):
    #         gauss2[row][col] = \
    #             (gauss[row - 2][col - 2] *  0 + gauss[row - 2][col - 1] *  0 + gauss[row - 2][col + 0] * -1 + gauss[row - 2][col + 1] *  0 + gauss[row - 2][col + 2] *  0 +
    #              gauss[row - 1][col - 2] *  0 + gauss[row - 1][col - 1] * -1 + gauss[row - 1][col + 0] * -2 + gauss[row - 1][col + 1] * -1 + gauss[row - 1][col + 2] *  0 +
    #              gauss[row + 0][col - 2] * -1 + gauss[row + 0][col - 1] * -2 + gauss[row + 0][col + 0] * 16 + gauss[row + 0][col + 1] * -2 + gauss[row + 0][col + 2] * -1 +
    #              gauss[row + 1][col - 2] *  0 + gauss[row + 1][col - 1] * -1 + gauss[row + 1][col + 0] * -2 + gauss[row + 1][col + 1] * -1 + gauss[row + 1][col + 2] *  0 +
    #              gauss[row + 2][col - 2] *  0 + gauss[row + 2][col - 1] *  0 + gauss[row + 2][col + 0] * -1 + gauss[row + 2][col + 1] *  0 + gauss[row + 2][col + 2] *  0)

    #del meanArr

    #cv2.imshow("gauss", gauss)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return gauss

###########################################################################
# 6. POST-FILTER THRESHOLDING (OPTIONAL)
###########################################################################
def postFilterThreshold(gauss, noiseThreshold, rows, cols):
    print 'post-filter thresholding'

    # drop all values below threshold - removes small noise
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            if gauss[row][col] < noiseThreshold:
                gauss[row][col] = 0

    #cv2.imshow("new gauss", gauss)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return gauss


###########################################################################
# 7. EDGE DETECTION
###########################################################################
def edgeDetection(gauss, rows, cols):
    print 'edge detection'

    gx = np.zeros((rows, cols))
    gy = np.zeros((rows, cols))

    # Sobel mask
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            # gx
            # -1 0 1
            # -2 0 2
            # -1 0 1
            gx[row][col] = (gauss[row - 1][col - 1] * -1 + gauss[row - 1][col] *  0 + gauss[row - 1][col + 1] *  1 +
                            gauss[row + 0][col - 1] * -2 + gauss[row + 0][col] *  0 + gauss[row + 0][col + 1] *  2 +
                            gauss[row + 1][col - 1] * -1 + gauss[row + 1][col] *  0 + gauss[row + 1][col + 1] *  1)
            # gy
            #  1  2  1
            #  0  0  0
            # -1 -2 -1
            gy[row][col] = (gauss[row - 1][col - 1] *  1 + gauss[row - 1][col] *  2 + gauss[row - 1][col + 1] *  1 +
                            gauss[row + 0][col - 1] *  0 + gauss[row + 0][col] *  0 + gauss[row + 0][col + 1] *  0 +
                            gauss[row + 1][col - 1] * -1 + gauss[row + 1][col] * -2 + gauss[row + 1][col + 1] * -1)

    del gauss

    # Canny mask
    # NOTE: we lose 1 row at the bottom and 1 column at the right
    # for row in range(1, rows-2):
    #     for col in range(1, cols-2):
    #         gx[row][col] = (gauss[row + 0][col + 0] * -1 + gauss[row + 0][col + 1] *  1 +
    #                         gauss[row + 1][col + 0] * -1 + gauss[row + 1][col + 1] *  1)
    #         gy[row][col] = (gauss[row + 0][col + 0] *  1 + gauss[row + 0][col + 1] *  1 +
    #                         gauss[row + 1][col + 0] * -1 + gauss[row + 1][col + 1] * -1)
    #
    # del gauss

    gradientMagnitude = np.zeros((rows, cols))

    # calculate gradient magnitude
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            gradientMagnitude[row][col] = np.sqrt( gx[row][col] ** 2 + gy[row][col] ** 2 )

    #print gradientMagnitude

    # remove garbage in outer 5 rows/cols
    for row in [0, 1, 2, 3, 4, rows-5, rows-4, rows-3, rows-2, rows-1]:
        for col in range(cols):
            gradientMagnitude[row][col] = 0
    for col in [0, 1, 2, 3, 4, cols-5, cols-4, cols-3, cols-2, cols-1]:
        for row in range(rows):
            gradientMagnitude[row][col] = 0

    plt.imshow(gradientMagnitude, cmap="gray")
    plt.savefig("gradientMagnitude3")

    #cv2.imshow("gradientMagnitude", gradientMagnitude)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return gradientMagnitude, gx, gy


###########################################################################
# 8. NONMAXIMA SUPPRESSION
###########################################################################
def nonmaximaSuppression(gradientMagnitude, gx, gy, rows, cols):
    print 'nonmaxima suppression'

    gradientAngle = np.zeros((rows, cols))

    # calculate gradient angle and make array holding these values for each pixel
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            # avoid divide by zero errors
            if gx[row][col] == 0:
                if gy[row][col] == 0:
                    gradientAngle[row][col] = 0
                elif gy[row][col] > 0:
                    gradientAngle[row][col] = float(90)
                else:
                    gradientAngle[row][col] = float(-90)
            else:
                # arctan formula
                gradientAngle[row][col] = math.degrees(np.arctan( gy[row][col] / gx[row][col] ))
                if gradientAngle[row][col] < 0:
                    gradientAngle[row][col] = 360 + gradientAngle[row][col]

    #print 'gradient angles\n' + str(gradientAngle)

    # divide array into the 4 sectors
    for row in range(2, rows-2):
        for col in range(2, cols-2):
            if ( gradientAngle[row][col] >= 0 and gradientAngle[row][col] < 22.5) or \
                (gradientAngle[row][col] >= 157.5 and gradientAngle[row][col] < 202.5) or \
                (gradientAngle[row][col] >= 337.5 and gradientAngle[row][col] < 360):
                gradientAngle[row][col] = 0  # set to sector 0
            elif ( gradientAngle[row][col] >= 22.5 and gradientAngle[row][col] < 67.5) or \
                (gradientAngle[row][col] >= 202.5 and gradientAngle[row][col] < 247.5):
                gradientAngle[row][col] = 1  # set to sector 1
            elif ( gradientAngle[row][col] >= 67.5 and gradientAngle[row][col] < 112.5) or \
                (gradientAngle[row][col] >= 247.5 and gradientAngle[row][col] < 292.5):
                gradientAngle[row][col] = 2  # set to sector 2
            else:
                gradientAngle[row][col] = 3  # set to sector 3


    nonmaxima = np.zeros((rows,cols))

    # NOTE: tried increasing size of nonmaxima window to 5x5 instead of 3x3, but bad results
    for row in range(2, rows - 2):
        for col in range(2, cols - 2):
            # if sector 0
            if gradientAngle[row][col] == 0:
                if ( #gradientMagnitude[row][col] < gradientMagnitude[row + 0][col - 2]) or \
                    (gradientMagnitude[row][col] < gradientMagnitude[row + 0][col - 1]) or
                    (gradientMagnitude[row][col] < gradientMagnitude[row + 0][col + 1])): # or \
                    #(gradientMagnitude[row][col] < gradientMagnitude[row + 0][col + 2]):
                    nonmaxima[row][col] = 0
                else:
                    nonmaxima[row][col] = gradientMagnitude[row][col]
            # if sector 1
            elif gradientAngle[row][col] == 1:
                if ( #gradientMagnitude[row][col] < gradientMagnitude[row - 2][col + 2]) or \
                    (gradientMagnitude[row][col] < gradientMagnitude[row - 1][col + 1]) or
                    (gradientMagnitude[row][col] < gradientMagnitude[row + 1][col - 1])): # or \
                    #(gradientMagnitude[row][col] < gradientMagnitude[row + 2][col - 2]):
                    nonmaxima[row][col] = 0
                else:
                    nonmaxima[row][col] = gradientMagnitude[row][col]
            # if sector 2
            elif gradientAngle[row][col] == 2:
                if ( #gradientMagnitude[row][col] < gradientMagnitude[row - 2][col + 0]) or \
                    (gradientMagnitude[row][col] < gradientMagnitude[row - 1][col + 0]) or
                    (gradientMagnitude[row][col] < gradientMagnitude[row + 1][col + 0])): # or \
                    #(gradientMagnitude[row][col] < gradientMagnitude[row + 2][col + 0]):
                    nonmaxima[row][col] = 0
                else:
                    nonmaxima[row][col] = gradientMagnitude[row][col]
            # if sector 3
            else:
                if ( #gradientMagnitude[row][col] < gradientMagnitude[row - 2][col - 2]) or \
                    (gradientMagnitude[row][col] < gradientMagnitude[row - 1][col - 1]) or
                    (gradientMagnitude[row][col] < gradientMagnitude[row + 1][col + 1])): # or \
                    #(gradientMagnitude[row][col] < gradientMagnitude[row + 2][col + 2]):
                    nonmaxima[row][col] = 0
                else:
                    nonmaxima[row][col] = gradientMagnitude[row][col]


    # remove garbage in outer 5 rows/cols
    for row in [0, 1, 2, 3, 4, rows-5, rows-4, rows-3, rows-2, rows-1]:
        for col in range(cols):
            nonmaxima[row][col] = 0
    for col in [0, 1, 2, 3, 4, cols-5, cols-4, cols-3, cols-2, cols-1]:
        for row in range(rows):
            nonmaxima[row][col] = 0

    #del gradientAngle
    #del gradientMagnitude

    #cv2.imshow("nonmaxima", nonmaxima)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return nonmaxima


###########################################################################
# 9. EDGE THRESHOLDING
###########################################################################
def edgeThresholding(nonmaxima, edgeThreshold, rows, cols):
    print 'edge thresholding'

    # display the histogram to pick a threshold
    #plt.hist(nonmaxima.ravel(), 256, [0,1])  # 256 bins, values in range 0-1
    #plt.show()

    # create new array of 0s and 1s, 1s = edges
    thresholdedArray = np.zeros((rows,cols))
    for row in range(rows):
        for col in range(cols):
            # if above threshold, bump up to 1
            if nonmaxima[row][col] > edgeThreshold:
                thresholdedArray[row][col] = 1

    #del nonmaxima
    # display new histogram after thresholding
    #plt.hist(thresholdedArray.ravel(), 256, [0,1])
    #plt.show()

    # remove garbage in outer 5 rows/cols
    for row in [0, 1, 2, 3, 4, rows-5, rows-4, rows-3, rows-2, rows-1]:
        for col in range(cols):
            thresholdedArray[row][col] = 0
    for col in [0, 1, 2, 3, 4, cols-5, cols-4, cols-3, cols-2, cols-1]:
        for row in range(rows):
            thresholdedArray[row][col] = 0

    plt.imshow(thresholdedArray, cmap="gray")
    plt.savefig("thresholdedArray3")


    # display new image after thresholding
    #cv2.imshow("thresholdedArray", thresholdedArray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return thresholdedArray


###########################################################################
# 10. HOUGH TRANSFORM
###########################################################################
def houghTransform(thresholdedArray, pBinSize, thetaBinSize, rows, cols):
    print 'hough transform'

    # p = xcos(t) + ysin(t), so max p value is x+y, or i+j, or rows+cols...
    #    (technically less, since sin(t) and cos(t) can't be 1 at the same time)
    # create bins for p and theta
    # how many p bins do you want?
    pBins = int((rows+cols)/pBinSize)
    # how many theta bins do you want?
    thetaBins = int(270/thetaBinSize)
    thetas = []
    # create array of all theta values to test
    # NOTE: these go from -90 to 270
    for i in range(thetaBins):
        thetas.append(-90 + (270 / thetaBins) * i)

    # NOTE: pBins*2
    accumulator = np.zeros((pBins*2, thetaBins))

    #print thetas

    # create an array of lines to match accumulator's [pBin][thetas[theta]] indices - each element is a list of points on that line
    lines = {}
    m = 0
    # build accumulator array and lines dictionary
    for row in range(rows):
        for col in range(cols):
            if thresholdedArray[row][col]:  # == 1
                for theta in range(len(thetas)):
                    # calculate p
                    p = col * np.cos(math.radians(thetas[theta])) + row * np.sin(math.radians(thetas[theta]))  # NOTE: col = x, row = y
                    # sort into pBins
                    pBin = int(round((p + (rows + cols)) / pBinSize)) - 1
                    # update accumulator
                    accumulator[pBin][theta] += 1
                    # add line to dictionary
                    if (pBin,thetas[theta]) not in lines.keys():
                        m += 1
                        #print 'new line - total: ' + str(m)
                        lines[(pBin, thetas[theta])] = [(col,row)]  # NOTE: col = x, row = y
                    else:
                        lines[(pBin, thetas[theta])].append((col,row))  # NOTE: col = x, row = y

    #del thresholdedArray
    #print [i for i in accumulator]
    #print len([value for key,value in lines.iteritems()])

    #plt.imshow(accumulator, cmap='gray')
    #plt.show()

    print 'dictionary size is: ' + str(m)

    return lines, accumulator


###########################################################################
# 11. FILTER LINES
###########################################################################
def filterLines(lines, accumulator, pBinSize, thetaBinSize, accumThreshold, shortLineThreshold, rows, cols):
    print 'filter lines'

    # create bins for p and theta
    # how many p bins do you want?
    pBins = int((rows + cols) / pBinSize)
    # how many theta bins do you want?
    thetaBins = int(270 / thetaBinSize)
    thetas = []
    # create array of all theta values to test
    # NOTE: these go from -90 to 270
    for i in range(thetaBins):
        thetas.append(-90 + (270 / thetaBins) * i)

    #print np.max(accumulator)
    cutoff = np.max(accumulator) * accumThreshold
    #print cutoff

    for row in range(pBins):
        for col in range(thetaBins):
            # remove all values less than the cutoff
            if accumulator[row][col] < cutoff:
                #print lines[(pVal,theta)]
                if (row, thetas[col]) in lines.keys():  # CHECK THIS************************
                    del lines[(row, thetas[col])]  # CHECK THIS******************************

    #print m
    print 'dictionary size is: ' + str(len(lines.keys()))

    # remove short lines from dictionary
    newLines = {}
    # 55 no windows
    for key,value in lines.iteritems():
        if len(value) > shortLineThreshold:
            newLines[key] = value

    del lines
    print 'dictionary size is: ' + str(len(newLines.keys()))

    #print 'accum shape ' + str(accumulator.shape)
    filteredArray = np.zeros((rows,cols))

    # rebuild the 2D array from the dictionary
    for key,value in newLines.iteritems():
        for point in value:
            filteredArray[point[1]][point[0]] = 1 # CHECK THIS***

    #cv2.imshow("filteredArray", filteredArray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return newLines, filteredArray

###########################################################################
# 12. LONGEST/SHORTEST LINE (OPTIONAL)
###########################################################################
def findLongestShortest(newLines, rows, cols):

    print 'longest/shortest line'

    mostPoints = 0
    leastPoints = rows*cols
    longestLine = []
    shortestLine = []
    for key,value in newLines.iteritems():
        # find strongest line
        if len(value) > mostPoints:
            mostPoints = len(value)
            longestLine = value
        # find weakest line
        if len(value) < leastPoints:
            leastPoints = len(value)
            shortestLine = value

    #print longestLine
    #print shortestLine

    longImg = np.zeros((rows, cols))
    for colRow in longestLine:
        longImg[colRow[1]][colRow[0]] = 1

    shortImg = np.zeros((rows, cols))
    for colRow in shortestLine:
        shortImg[colRow[1]][colRow[0]] = 1

    #del longestLine
    #del shortestLine
    #print longImg

    cv2.imshow("longImg", longImg)
    cv2.imshow("shortImg", shortImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #del longImg
    #del shortImg


###########################################################################
# 13. FIND INTERSECTIONS OF PARALLEL LINES TO FIND PARALLELOGRAMS
###########################################################################
def findIntersections(newLines, pBinSize, minDist, angleTolerance, minAngle, minPoints, rows, cols):
    print 'find intersections of parallel lines to find parallelograms'

    allLines = []
    allPoints = []
    linePairs12 = []

    loops = len(newLines.keys())
    loop = 0

    """TODO: current implementation is O(n^4 * m * p) time which sounds terrible, though dynamic programming and if-statements
    prevent most wasted operations, so speed is actually very good. can improve though by pairing up parallel lines in a
    separate O(n^2) loop prior to this to remove the need for so many if-statements below"""
    # pick 1 line
    for key1, value1 in newLines.iteritems():
        loop += 1
        print str(loop) + '/' + str(loops)
        # pick another line
        for key2, value2 in newLines.iteritems():
            # must be different lines
            if key1 != key2:
                # check for parallel
                if abs(key1[1] - key2[1]) < angleTolerance:
                    # find pBin value from keys
                    # pBin = int(round(p / pBinSize)) - 1
                    newP1 = (key1[0] + 1) * pBinSize - (rows+cols)
                    newP2 = (key2[0] + 1) * pBinSize - (rows+cols)
                    # check they are minimum distance apart
                    if abs(newP1 - newP2) > minDist:
                        # check list to see if we've compared these lines already
                        if [min(key1, key2), max(key1, key2)] not in linePairs12:
                            # if not in list, add to list in sorted order and proceed
                            linePairs12.append([min(key1, key2), max(key1, key2)])
                            linePairs34 = []
                            # pick 1 line
                            for key3, value3 in newLines.iteritems():
                                # pick another line
                                for key4, value4 in newLines.iteritems():
                                    # must be different lines
                                    if key3 != key4:
                                        # make sure all 4 lines are not parallel
                                        if key1[1] != key3[1]:
                                            # check for parallel
                                            if abs(key3[1] - key4[1]) < angleTolerance:
                                                # pBin = int(round(p / pBinSize)) - 1
                                                newP3 = (key3[0] + 1) * pBinSize - (rows+cols)
                                                newP4 = (key4[0] + 1) * pBinSize - (rows+cols)
                                                # check they are minimum distance apart
                                                if abs(key3[0] - key4[0]) > minDist:
                                                    # check they are minimum angle apart
                                                    if abs(key1[1] - key3[1]) > minAngle:
                                                        if [min(key3, key4), max(key3, key4)] not in linePairs34:
                                                            linePairs34.append([min(key3, key4), max(key3, key4)])

                                                            # find (x1,y1) - intersection of key1 and key3
                                                            y1 = ( (newP1 / np.cos(math.radians(key1[1]))) - (newP3 / np.cos(math.radians(key3[1]))) ) / ( np.tan(math.radians(key1[1])) - np.tan(math.radians(key3[1])) )
                                                            x1 = ( newP1 - (y1 * np.sin(math.radians(key1[1]))) ) / np.cos(math.radians(key1[1]))

                                                            # find (x2,y2) - intersection of key3 and key2
                                                            y2 = ((newP3 / np.cos(math.radians(key3[1]))) - (newP2 / np.cos(math.radians(key2[1])))) / (np.tan(math.radians(key3[1])) - np.tan(math.radians(key2[1])))
                                                            x2 = (newP3 - (y2 * np.sin(math.radians(key3[1])))) / np.cos(math.radians(key3[1]))

                                                            # find (x3,y3) - intersection of key2 and key4
                                                            y3 = ((newP2 / np.cos(math.radians(key2[1]))) - (newP4 / np.cos(math.radians(key4[1])))) / (np.tan(math.radians(key2[1])) - np.tan(math.radians(key4[1])))
                                                            x3 = (newP2 - (y3 * np.sin(math.radians(key2[1])))) / np.cos(math.radians(key2[1]))

                                                            # find x1,y1 - intersection of key4 and key1
                                                            y4 = ((newP4 / np.cos(math.radians(key4[1]))) - (newP1 / np.cos(math.radians(key1[1])))) / (np.tan(math.radians(key4[1])) - np.tan(math.radians(key1[1])))
                                                            x4 = (newP4 - (y4 * np.sin(math.radians(key4[1])))) / np.cos(math.radians(key4[1]))

                                                            # make sure no points are out of range of the image
                                                            if y1 > 0 and y1 < rows and y2 > 0 and y2 < rows and y3 > 0 and y3 < rows and y4 > 0 and y4 < rows and x1 > 0 and x1 < cols and x2 > 0 and x2 < cols and x3 > 0 and x3 < cols and x4 > 0 and x4 < cols:

                                                                # check number of pixels on line1 between x1,y1 and x4,y4
                                                                counter = 0
                                                                for eachPoint in value1:
                                                                    if eachPoint[0] >= min(x1, x4) and eachPoint[0] <= max(x1, x4) and eachPoint[1] >= min(y1, y4) and eachPoint[1] <= max(y1, y4):
                                                                        counter += 1
                                                                if counter > minPoints:

                                                                    # check number of pixels on line2 between x2,y2 and x3,y3
                                                                    counter = 0
                                                                    for eachPoint in value2:
                                                                        if eachPoint[0] >= min(x2, x3) and eachPoint[0] <= max(x2, x3) and eachPoint[1] >= min(y2, y3) and eachPoint[1] <= max(y2, y3):
                                                                            counter += 1
                                                                    if counter > minPoints:

                                                                        # check number of pixels on line3 between x1,y1 and x2,y2
                                                                        counter = 0
                                                                        for eachPoint in value3:
                                                                            if eachPoint[0] >= min(x1, x2) and eachPoint[0] <= max(x1, x2) and eachPoint[1] >= min(y1, y2) and eachPoint[1] <= max(y1, y2):
                                                                                counter += 1
                                                                        if counter > minPoints:

                                                                            # check number of pixels on line4 between x3,y3 and x4,y4
                                                                            counter = 0
                                                                            for eachPoint in value4:
                                                                                if eachPoint[0] >= min(x3, x4) and eachPoint[0] <= max(x3, x4) and eachPoint[1] >= min(y3, y4) and eachPoint[1] <= max(y3, y4):
                                                                                    counter += 1
                                                                            if counter > minPoints:

                                                                                try:
                                                                                    # add lines to dictionary
                                                                                    allLines.append([(int(round(x1)),int(round(y1))),(int(round(x2)),int(round(y2))),(int(round(x3)),int(round(y3))),(int(round(x4)),int(round(y4)))])
                                                                                    print 'allLines length: ' + str(len(allLines))
                                                                                except:
                                                                                    # just in case x or y is infinity due to divide by zero error
                                                                                    pass

    print allLines
    print len(allLines)

    return allLines


###########################################################################
# 14. DRAW PARALLEL LINES
###########################################################################
def drawParallelLines(file, filteredArray, allLines, originalRows, originalCols, rows, cols):
    print 'draw parallel lines'

    # stretch array to 3x the # of columns
    filteredArray = np.uint8(filteredArray)
    finalArray = cv2.cvtColor(filteredArray,cv2.COLOR_GRAY2RGB)

    # reopen file for original image
    file = file[:-4] + ".raw"

    pic = open(file, 'rb')
    f = np.fromfile(pic, dtype=np.uint8)
    pic.close()

    img = f.reshape((originalRows,originalCols*3))

    #print img

    # convert image to usable array
    for row in range(rows):
        for col in range(cols):
            for rgb in range(3):
                # TODO: [col*3*4+rgb] below gives unexpected results (check cv2.COLOR_GRAY2RGB conversion) - leaving image grayscale for now
                finalArray[row][col][rgb] = img[row*4][col*3*4]


    # draw the lines between points
    for i in range(len(allLines)):
        try:
            finalArray = cv2.line(finalArray, (allLines[i][0][0], allLines[i][0][1]), (allLines[i][1][0], allLines[i][1][1]), (255, 0, 0), 1)
            finalArray = cv2.line(finalArray, (allLines[i][1][0], allLines[i][1][1]), (allLines[i][2][0], allLines[i][2][1]), (255, 0, 0), 1)
            finalArray = cv2.line(finalArray, (allLines[i][2][0], allLines[i][2][1]), (allLines[i][3][0], allLines[i][3][1]), (255, 0, 0), 1)
            finalArray = cv2.line(finalArray, (allLines[i][3][0], allLines[i][3][1]), (allLines[i][0][0], allLines[i][0][1]), (255, 0, 0), 1)
        except:
            pass

    cv2.imshow('finalArray', finalArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return finalArray



###########################################################################
# MAIN
###########################################################################
def main():

    # TODO: pass file as an arg in command line, need to update section 1 to remove .raw
    file = "C:\Directory\Filename.jpg"

    # initialize variables - ADJUST THESE VALUES
    noiseThreshold = 0.05      # drop very low values to 0 prior to edge detection to remove noise
    edgeThreshold = 0.14       # minimum edge strength required after nonmaxima suppression
    pBinSize = 1               # normal to line distance from origin, used in accumulator
    thetaBinSize = 2           # normal to line degrees from origin, used in accumulator
    accumThreshold = 0.23      # threshold the accumulator and remove low key:value pairs from lines dictionary
    shortLineThreshold = 23    # minimum number of edge pixels on a line for line to be acceptable
    findLongShort = False      # display longest and shortest lines found
    minDist = 30               # minimum Euclidean distance for acceptable intersection points
    angleTolerance = 8         # maximum angle tolerance (in degrees) for 'parallel' lines
    minAngle = 75              # minimum angle of intersecting lines. the closer to 90 degrees, the more 'square' the parallelogram
    minPoints = 22             # minimum number of edge pixels on a line between the (x,y) intersections with 2 other parallel lines

    # open and size the file
    img, originalRows, originalCols = openAndSizeFile(file)

    # convert it to gray
    gray = convertToGray(img, originalRows, originalCols)

    # resize image for faster processing, thinner edges
    smallArr, rows, cols = resizeFile(gray, originalRows, originalCols)

    # mean filter
    meanArr = meanFilter(smallArr, rows, cols)

    # Gaussian/Laplacian/LoG filter
    gauss = gaussianFilter(meanArr, rows, cols)

    # post-filter thresholding
    gauss = postFilterThreshold(gauss, noiseThreshold, rows, cols)

    # edge detection - Sobel or Canny
    gradientMagnitude, gx, gy = edgeDetection(gauss, rows, cols)

    # nonmaxima suppression
    nonmaxima = nonmaximaSuppression(gradientMagnitude, gx, gy, rows, cols)

    # edge thresholding
    thresholdedArray = edgeThresholding(nonmaxima, edgeThreshold, rows, cols)

    # Hough transform
    lines, accumulator = houghTransform(thresholdedArray, pBinSize, thetaBinSize, rows, cols)

    # filter lines
    newLines, filteredArray = filterLines(lines, accumulator, pBinSize, thetaBinSize, accumThreshold, shortLineThreshold, rows, cols)

    # find longest and shortest lines
    if findLongShort:
        findLongestShortest(newLines, rows, cols)

    # find intersection points
    allLines = findIntersections(newLines, pBinSize, minDist, angleTolerance, minAngle, minPoints, rows, cols)

    # draw the parallelograms
    finalArray = drawParallelLines(file, filteredArray, allLines, originalRows, originalCols, rows, cols)


# call main()
if __name__ == "__main__":
    print('Object Identification: Parallelograms - by Jesse Lew\n')
    main()
