import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import rotate
from skimage.measure import block_reduce
from skimage.segmentation import clear_border
from skimage import data, filters, measure, morphology
import math
import pandas as pd
import imutils
import csv
import os
import tifffile

#load images: Brightfield(BF), Bluorescent image (FF)
OME_directory = r'G:\mju cloud\OneDrive\☆UBC\☆Research\data\Export'
OME_base_name = '2023_01_22_WellE23_ChannelBF_20XELWD,CY3_20XELWD,CY5_20XELWD,DAPI_20XELWD_Seq0003.tif'
filename_OME= os.path.join(OME_directory, OME_base_name)
print("File path:", filename_OME)
img1_BF = tifffile.imread(filename_OME, key=0)
#img1_BF = cv2.cvtColor(img1_BF,cv2.COLOR_BGR2GRAY)
img1_FF = tifffile.imread(filename_OME, key=3)
img1_FF = cv2.normalize(img1_FF, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#img1_FF = cv2.cvtColor(img1_FF,cv2.COLOR_BGR2GRAY)

class JoyceSegment(SegmentNanowell):
    # Function to display the coordinates of
    # of the points clicked on the image
    # This came from: https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/ and
    # https://learnopencv.com/mouse-and-trackbar-in-opencv-gui/ 
    
    def render_microscopy_image(img: np.ndarray, top_left: list, bot_right:list, fill_color: np.ndarray) -> np.ndarray:

        boundaries = np.zeros_like(img, np.uint8) # Create a numpy array of zeros of the same size as 'img'
        for tl, br in zip(top_left, bot_right):
            curr_tl = (tl.y, tl.x) 
            curr_br = (br.y, br.x) 
            
            cv2.rectangle(boundaries, curr_tl, curr_br, fill_color.tolist(), -1)
        aspect = img.shape[0] / img.shape[1]
        out = cv2.resize(img, (int(700*aspect), 700)) 
        boundaries = cv2.resize(boundaries, (int(700*aspect), 700)) 
        alpha = 0.5
        mask = boundaries.astype(bool)
        out[mask] = cv2.addWeighted(out, alpha, boundaries, 1 - alpha, 0)[mask]
        return out
        ## Test using only pixel values (NOT CONVERTED TO UM)

    @staticmethod
    def segment_image(image: MicroscopyImage) -> None: 
        # Create lists to store the bounding box coordinates
        top_left= None # For tile image
        bot_right= None # For tile image
        nwell_top_left = []
        nwell_bot_right = []
        # Important constants to be defined
        px_to_um = 0.37 # (um/px) Pixel-to-um conversion for images taken at 20x magnification
        nanowell_width = 92 # Units: um 87.1; 90.28
        nanowell_height = 89 # Units: um (83.3 um) 85 um
        pitch_x = 98.29 # Nanowell pitch = 97 um (along x-axis); 97.4 
                                    # Round converts data type to int
        pitch_y = 94.66 # Nanowell pitch = 96 um (along y-axis); 95.95
        
        fill_color = np.random.choice(range(256), size=3) # Numpy array of 3 elements, each representing a value for 'BGR'
        
        def calculate_nanowells():
            nonlocal nwell_top_left
            nonlocal nwell_bot_right

            num_nanowells_x = round((bot_right.x - top_left.x) / um_to_px(pitch_x))
            num_nanowells_y = round((bot_right.y - top_left.y) / um_to_px(pitch_y))

            nwell_top_left = []
            nwell_bot_right = []

            base_x = top_left.x
            base_y = top_left.y
            
            nw_width = um_to_px(nanowell_width)
            nw_height = um_to_px(nanowell_height)
            nw_pitch_x = um_to_px(pitch_x)
            nw_pitch_y = um_to_px(pitch_y)

            for j in range(num_nanowells_y):
                for i in range(num_nanowells_x):
                    tl_x = base_x + (i * nw_pitch_x)
                    tl_y = base_y + (j * nw_pitch_y)

                    br_x = tl_x + nw_width
                    br_y = tl_y + nw_height
                    # cv2 has reversed coordinates... these will be compatible outside of cv2
                    nwell_top_left.append(PxCoord(tl_y, tl_x))
                    nwell_bot_right.append(PxCoord(br_y, br_x))
            return


        def um_to_px(x):
            x = round(x / px_to_um)
            return x        

        def click_event(event, x, y, flags, params):
            nonlocal top_left
            nonlocal bot_right

            # checking for left mouse clicks
            if event == cv2.EVENT_LBUTTONDOWN:
                top_left = PxCoord(x,y) # Add pixel coordinates to list
       
                # displaying the coordinates
                # on the image window
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(x) + ',' +
                            str(y), (x,y), font,
                            2, (255, 255, 255), 3)
                cv2.imshow("Nanowells", img)

            if event == cv2.EVENT_RBUTTONDOWN:
                bot_right = PxCoord(x,y) # Add pixel coordinates to list

                 # displaying the coordinates
                # on the image window
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(x) + ',' +
                            str(y), (x,y), font,
                            2, (255, 255, 255), 3)
                cv2.imshow("Nanowells", img)    

            if top_left and bot_right:
                calculate_nanowells()
                curr_mask = JoyceSegment.render_microscopy_image(img, nwell_top_left, nwell_bot_right, fill_color)
                cv2.imshow("Nanowells", curr_mask)

        # Read the image
        img = image.image

        # Custom window
        # This does not maintain the aspect ratio when the window is resized
        cv2.namedWindow("Nanowells", cv2.WINDOW_NORMAL)
        cv2.imshow("Nanowells", img)

        # Setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback("Nanowells", click_event)
    
        # Wait for a key to be pressed to exit
        cv2.waitKey(0)
        
        # Close the window
        cv2.destroyAllWindows()
        
        calculate_nanowells()
        for i in range(len(nwell_bot_right)):
            nwell_x_stage = image.px_scale*nwell_top_left[i].x + image.stage_coord.x
            nwell_y_stage = image.px_scale*nwell_top_left[i].y + image.stage_coord.y
            nwell_coord = StageCoord(nwell_x_stage, nwell_y_stage, image.stage_coord.z)
            
            curr_nwell = Nanowell(nwell_top_left[i], nwell_bot_right[i], nwell_coord, [])
            image.nanowells.append(curr_nwell)
        
        return 

    @staticmethod
    def segment_images(images: List[MicroscopyImage]) -> None:
        for image in images:
            JoyceSegment.segment_image(image)
        
        return
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
