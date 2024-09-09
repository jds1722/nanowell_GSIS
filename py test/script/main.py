import sys
import math
import cv2 as cv
import numpy as np
from PIL import Image
import tifffile
import timeit

def main(argv):

    filename = 'nd_crop14-1.tif'

    start = timeit.default_timer()

    grid = 14
    #groupThreshold = 50 #50 for small sizes -> 3x3 -4x4 , 55 for 6x6
    groupThreshold = 90
    #adjust based on the picturesize #tune this for grouping TUNE THIS UNTIL FINALPOINTS = (GRID+1)*(GRID+1)
    columnSize = grid +1

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    mylist = []
    loaded, mylist = cv.imreadmulti(mats=mylist, filename = filename, flags=cv.IMREAD_ANYCOLOR)

    # alternative usage
    # loaded,mylist = cv2.imreadmulti(mats = mylist, start =0, count = 2, filename = "2page.tiff", flags = cv2.IMREAD_ANYCOLOR )

    cv.imshow("mylist[0]", mylist[0])
    cv.imshow("mylist[1]", mylist[1])


    #image = cv.imread(cv.samples.findFile(filename),cv.IMREAD_GRAYSCALE)
    #image = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    #image = cv.imreadmulti(filename,1,cv.IMREAD_GRAYSCALE)
    image_sharp = cv.filter2D(mylist[0], ddepth=-1, kernel=kernel)
    cv.imshow('AV CV- Winter Wonder Sharpened', image_sharp)

    #src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    # src= cv.filter2D(src, ddepth=-1, kernel=kernel)
    # # Check if image is loaded fine
    # if src is None:
    #     print('Error opening image!')
    #     print('Usage: hough_lines.py [image_name -- default ' + filename + '] \n')
    #     return -1

    #edge detection
    dst = cv.Canny(image_sharp, 50, 200 , None, 3)
    cv.imshow("dst",dst)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    img_inter4 = cv.resize(cdstP, (800, 800), interpolation=cv.INTER_NEAREST)

    cv.imshow("lines", img_inter4)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 170, None, 250, 25) #SECOND LAST VAR: SMALLER = SLOWER BUT MORE ACCURATE
    # work for test11
    #linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 130, None, 150, 30)
    #work for test9
    #linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 120, 25)
    # work for test2
    #linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 200, None, 150, 40)
    print("Hough -> done!")
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)
    h_lines,v_lines = segment_lines(linesP,10)

    print("Find Lines -> done!")
    img_inter3= cv.resize(cdstP, (800, 800), interpolation=cv.INTER_NEAREST)

    cv.imshow("line", img_inter3)
    cv.waitKey(0)

    # find the line intersection points
    Px = []
    Py = []
    for h_line in h_lines:
        for v_line in v_lines:
            px, py = find_intersection(h_line, v_line)
            px = int(px)
            py = int(py)
            Px.append(px)
            Py.append(py)

    print("Find intersections -> done!")


    pairedList = list(zip(Px,Py))

    # printing original list
    #print("The original list is : " + str(pairedList))

    tempsort= []
    masterList = []
    pairedList = sorted(pairedList, key=lambda a: a[0])
    #print("sorted pairlist", pairedList)
    while len(pairedList) != 0:
        tempsort = []
        for ele in pairedList:
            if (abs(pairedList[0][0] - ele[0]) < groupThreshold):
                tempsort.append(ele)
            else:
                break

        tempsort = sorted(tempsort, key=lambda a: a[1])
        #print("tempsort", tempsort)
        for remove in tempsort:
            pairedList.remove(remove)
        means(tempsort, groupThreshold, masterList)

    #print(masterList)

    # while len(pairedList)!=0:
    #
    #     for pair in pairedList:
    #         #group and find average
    #         a = abs(pairedList[0][0] - pair[0])
    #         b= abs(pair[1] - pairedList[0][1])
    #         if (a <= groupThreshold) & (b <= groupThreshold):
    #             temp.append(pair)
    #         else:
    #             meanx = sum(elt[0] for elt in temp) / len(temp)
    #             meany = sum(elt[1] for elt in temp) / len(temp)
    #             finalPoints.append([(meanx,meany)])
    #
    #     for remove in temp:
    #         pairedList.remove(remove)
    #     temp = []


    # # Group Adjacent Coordinates
    # # Using product() + groupby() + list comprehension
    # man_tups = [sorted(sub) for sub in product(pairedList, repeat=2)
    #             if Manhattan(*sub) < groupThreshold]
    # print('mantup is done')
    #
    # res_dict = {ele: {ele} for ele in pairedList}
    # for tup1, tup2 in man_tups:
    #     res_dict[tup1] |= res_dict[tup2]
    #     res_dict[tup2] = res_dict[tup1]
    # print('for tup1, tup2 in man_tups')
    #
    #
    # res = [[*next(val)] for key, val in groupby(
    #     sorted(res_dict.values(), key=id), id)]
    # print('es = [[*next(val)] for key, val in groupby(sorted(res_dict.values(), key=id), id)]')
    #
    # finalPoints = []
    # for i in res:
    #     finalPoints.append(tuple(mean(i, axis=0)))

    #print('the final points is',finalPoints)
    #print('the final length is', len(finalPoints))
    finalPoints = masterList
    stop = timeit.default_timer()
    print('Grouping intersections -> done!')

    #cleaning



    # printing result
    #print("The grouped elements : " + str(res))
    intersectsimg = image_sharp.copy()
    for cx, cy in zip(Px, Py):
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        #color = np.random.randint(0, 255, 3).tolist()  # random colors
        color = (0,255,0)
        cv.circle(intersectsimg, (cx, cy), radius=2, color=color, thickness=-1)  # -1: filled circle

    img_inter = cv.resize(intersectsimg, (800, 800), interpolation=cv.INTER_NEAREST)
    cv.imshow("Intersections", img_inter)
    cv.waitKey(0)

    key = cv.waitKey(0)
    #final points
    fpoints = image_sharp.copy()
    i=0
    for point in masterList:
        x = point
        a = int(x[0][0])
        b = int(x[0][1])
        point=np.round(point).astype(int)
        #print(point)
        color = (0, 255, 0)
        cv.circle(fpoints, (a,b), radius=10, color=color, thickness=-1)  # -1: filled circle
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(fpoints, str(i), (a-10,b-10), font, 1, (255, 255, 255), 2, cv.LINE_AA)
        i = i+1


    cv.imshow("Intersections", fpoints)
    img_inter2 = cv.resize(fpoints, (800, 800), interpolation=cv.INTER_NEAREST)

    cv.imshow("fpoint", img_inter2)
    cv.waitKey(0)

    #finalPoints = np.round(finalPoints).astype(int)
    #print(finalPoints)

    #finalpoints = sorted(finalPoints,key=lambda element: (element[0]))
    finalpoints = finalPoints
    x=0
    #sort
   # for i in range(columnSize):
       # finalpoints[0+columnSize*(i):columnSize*(i+1)] = sorted(finalpoints[0+columnSize*(i):columnSize*(i+1)], key=lambda element: (element[1]))


    # temp = []
    # skip = False
    # groupSize = groupThreshold
    #
    # for i in range(len(finalpoints)):
    #     for checkPoint in finalpoints:
    #         if (abs(finalpoints[i][0][0] - checkPoint[0][0]) <groupThreshold)& (abs(finalpoints[i][0][1] - checkPoint[0][1]) <groupThreshold) & (finalpoints[i]!=checkPoint):
    #             temp.append(finalpoints[i])
    #             print('point:',finalpoints[i],'checkpoint:',checkPoint)
    #
    #
    # for i in range(len(temp)-1):
    #     finalpoints.remove(temp[i])
    #
    #     print(i,'removed:',temp[i])



        # finalpoints[0:columnSize] = sorted(finalpoints[0:columnSize], key=lambda element: (element[1]))
        # finalpoints[columnSize:columnSize*2] = sorted(finalpoints[columnSize:columnSize*2], key=lambda element: (element[1]))
        # finalpoints[columnSize*2:columnSize*3] = sorted(finalpoints[columnSize*2:columnSize*3], key=lambda element: (element[1]))
        # finalpoints[columnSize*3:columnSize*4] = sorted(finalpoints[columnSize*3:columnSize*4] , key=lambda element: (element[1]))

    # cv.imshow("Intersections", intersectsimg)
    # img_inter = cv.resize(intersectsimg, (600, 600), interpolation=cv.INTER_NEAREST)
    #
    #
    # cv.imshow("Intersections", img_inter)
    # cv.imwrite('fpoint.png', fpoints)
    # cv.waitKey(0)
    #
    # cv.imshow("Intersections", fpoints)
    # img_inter2 = cv.resize(fpoints, (600, 600), interpolation=cv.INTER_NEAREST)
    #
    # cv.imshow("fpoint", img_inter2)
    # cv.waitKey(0)
    # cv.imwrite('intersections.png', img_inter)

    cropList = []
    cropList2 = []

    for x in range(columnSize*columnSize-columnSize):
        if x% columnSize != grid :
            cropList.append(finalpoints[x])
            cropList2.append(finalpoints[x+columnSize+1])

    print('report: intersection points:', len(finalpoints))
    # print('CropList: ',cropList)
    # print("final sorted:",finalpoints)


    # Opens a image in RGB mode
    im = Image.open(filename)
    img = cv.imreadmulti(filename)
    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    #print('the number is ',Px[0],',', Py[0],',', px[10],',',Py[10])
    #im.seek(1)
    # Shows the image in image viewer
    #im1.show()
    y = 0
    c= 0
    i=0
    x=0
    CList = []
    for x in range(len(cropList)):
        left = cropList[x][0][0]
        top = cropList[x][0][1]
        right = cropList2[x][0][0]
        bottom = cropList2[x][0][1]
        #print('well point',left, top, right, bottom)
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        im1 = im.crop((left, top, right, bottom))

        #imsum = np.array.append(im1)
        # imageName = 'well(' + str(c)+ ',' + str(y) + ').tif'
        # im1.save(imageName,save_all=True)

        for img in mylist:
            cropped_img = img[top:bottom , left:right]
            CList.append(cropped_img)

        y=y+1
        if y % grid == 0:
            c =c+1
            y=0
        i=i+1

    for i in range(grid*grid):
        testList = []
        testList.append(CList[0])
        testList.append(CList[1])
        testList.append(CList[2])
        testList.append(CList[3])
        testarray = np.array(testList)

        Name = filename+'-'+str(i) +'.tif'
        tifffile.imwrite(Name,data = testarray,ome=True,metadata={'Channel':{'Name':['BF_20X','CY3_20X','EGFP_20X','CY5_20X']}})


        CList.remove(CList[0])
        CList.remove(CList[0])
        CList.remove(CList[0])
        CList.remove(CList[0])



    print('Time: ', stop - start)
    print('report: cropped ->',i, 'images', 'intersection points:', len(finalpoints))
    return 0

def Manhattan(tup1, tup2):
    return abs(tup1[0] - tup2[0]) + abs(tup1[1] - tup2[1])

def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def means(pairedList, group, finalpoints):
    while (len(pairedList) != 0):
        temp = []
        for pair in pairedList:
            # group and find average
            a = abs(pairedList[0][0] - pair[0])
            b = abs(pair[1] - pairedList[0][1])
            if (a <= group) & (b <= group):
                temp.append(pair)

        meanx = sum(elt[0] for elt in temp) / len(temp)
        meany = sum(elt[1] for elt in temp) / len(temp)
        finalpoints.append([(meanx, meany)])
        for remove in temp:
            pairedList.remove(remove)

if __name__ == "__main__":
    main(sys.argv[1:])