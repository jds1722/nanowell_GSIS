import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import os

#a peek at the result
def plt_fun(arr, title, size=(10,10)):
    plt.figure(figsize=size)
    plt.imshow(arr)
    plt.title(title)
    #plt.colorbar()
    plt.show()

#plot the 2D grid overlapped with edges of each nanowell unit
def nano_mask(start_x, start_y, Unit_hight, Unit_width, m, n): 
    ver_lines=[]
    hor_lines=[]
    mask_lines=[]
    for i in range (m+1):
        ver_lines.append(start_x+i*Unit_width)

    for i in range (n+1):
        hor_lines.append(start_y+i*Unit_hight)

    right_bottom_x=start_x+m*Unit_width
    right_bottom_y=start_y+n*Unit_hight

    for i in range (m+1):
        mask_lines.append([ver_lines[i], start_y, ver_lines[i], right_bottom_y])   
        
    for i in range (n+1):
        mask_lines.append([start_x, hor_lines[i], right_bottom_x, hor_lines[i]])   
    return mask_lines


#----------------------------------
#build a lookup table for unit address. Unit No.0 is top left one. 
table=[] # a list
count=0
for i in range(N*N):
    table.append([(i%M)*unit_width, (i//M)*unit_hight])

#draw a square for wanted units. Need to use the looup table
def draw_unit(img, cell_list):
    fig = plt.figure(figsize=(17, 17))
    fig.add_subplot(1, 1, 1)
    for i in range(len(cell_list)):
        x=int(table[cell_list[i]][0])
        y=int(table[cell_list[i]][1])
        cv2.rectangle(img, pt1=(x,y), pt2=(x+unit_width,y+unit_hight), color=(255,0,0), thickness=10)
    plt.imshow(img, vmin=0, vmax=100)

    ###fig.add_subplot(2, 1, 2)
    #whiteblankimage = 255 * np.ones(shape=(img.shape[0],img.shape[1], 3), dtype=np.uint8)
    #for i in range(len(cell_list)):
    #    x=int(table[cell_list[i]][0])
    #    y=int(table[cell_list[i]][1])
    #    cv2.rectangle(whiteblankimage, pt1=(x,y), pt2=(x+unit_width,y+unit_hight), color=(0,0,0), thickness=20)
    #    cv2.putText(whiteblankimage, text=str(cell_list[i]), org=(x-50,y+120),fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,0,0),thickness=10, lineType=cv2.LINE_AA)
    #plt.imshow(whiteblankimage)
    #plt.show()### 