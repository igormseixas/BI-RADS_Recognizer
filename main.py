import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys


#Read the image and verify if is it loaded correctly.
image = cv.imread("d_left_mlo (44).png", cv.IMREAD_COLOR)
if image is None:
    sys.exit("Could not read the image.")

#Get a copy of the original image for futher use.
copyImage = image.copy()

#Inicializate a select image.
selectImage = None

#Get original image width and height for futher use.
originalWidth = image.shape[1]
originalHeight = image.shape[0]

#Mouse functions.
def mouse_events(event, x, y, flags, param):
    #print(event)
    #print(flags)
    global image
    global copyImage
    global selectImage
    
    if event == cv.EVENT_LBUTTONDBLCLK:
        #Clears image before making a new sub-rect.
        image = copyImage.copy()

        #Crop the sub-rect from the image
        region_size = 128 #Region size may change to 128, 64 and 32.
        overlay = image[y-(region_size//2):y+(region_size//2), x-(region_size//2):x+(region_size//2)]
        #Copy information to a selected image.
        selectImage = overlay.copy()
        #Define a blue rectangle in the same shape as previously selected.
        blue_rect = np.full(overlay.shape, (255,0,0), dtype=np.uint8) #Build rectangle and set the blue color (255,0,0)

        #Add the rectangle to the selected and previously cut area of an image. Scale the transparency.
        cv.addWeighted(overlay, 0.5, blue_rect, 0.5, 1.0)
        transparency=0.7 #Greater the value, greater the transparency is.
        gamma=10.0 #Gamma of the selected area, more gamma more white will me added.
        select = cv.addWeighted(overlay, transparency, blue_rect, 1-transparency, gamma)
        image[y-(region_size//2):y+(region_size//2), x-(region_size//2):x+(region_size//2)] = select
        
        #Show area of interest in another window.
        cv.namedWindow("Selected image window", cv.WINDOW_AUTOSIZE)
        cv.imshow("Selected image window", selectImage)
    
    if event == cv.EVENT_MOUSEWHEEL: 
        #Sign of the flag shows direction of mousewheel.
        if flags > 0:
            #scroll up
            #cv.moveWindow("Display window", 0, 0)
            image = cv.resize(image,None,fx=1.1, fy=1.1, interpolation = cv.INTER_AREA)
        else:
            print(originalWidth < image.shape[1])
            print(originalHeight < image.shape[0])
            #scroll down
            if originalWidth < image.shape[1] and originalHeight < image.shape[0]:
                image = cv.resize(image,None,fx=0.9091, fy=0.9091, interpolation = cv.INTER_AREA)
        
        print(image.shape)

def changeResolution(newResolutionImage, newWidth, newHeight):
    
    #Dimenstion we wanted to be change.
    dimension= (newWidth,newHeight)   
    #First set the resolution to the image.
    newResolutionImage = cv.resize(newResolutionImage,dimension, interpolation = cv.INTER_AREA)   
    #Return the image to your original size with the new resolution.
    #newResolutionImage = cv.resize(newResolutionImage, (originalWidth, originalHeight), interpolation = cv.INTER_AREA)

    return newResolutionImage

def changeGrayScale(newGrayScale):
    global image
    
    #Convert image to Gray.
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    newGrayScale = newGrayScale-1

    for i in range(0,grayImage.shape[0]):
        for j in range(0,grayImage.shape[1]):
            #To quantize each channel into N levels (assuming the input is in the range [0,255]. 
            # You can also use floor, which is more suitable in some cases. 
            # This leads to N^3 different colors. 
            # For example with N=8 you get 512 unique RGB colors.
            #round(img*(N/255))*(255/N)
            grayImage[i,j] = round(grayImage[i,j]*(newGrayScale/255))*(255/newGrayScale) #Uniform quantization.

    cv.namedWindow("Test window", cv.WINDOW_AUTOSIZE)
    cv.imshow("Test window", grayImage)
    print(grayImage[100,100])

cv.namedWindow("Display window", cv.WINDOW_AUTOSIZE)
cv.setMouseCallback("Display window", mouse_events)

print(image.shape)

#Teste Plot LIB
#plt.imshow(image)
#plt.show()

#Show image, ESQ for exit.
while(1):
    cv.imshow("Display window", image)
    #cv.resizeWindow("Display window", originalWidth, originalHeight)
    k = cv.waitKey(1) & 0XFF
    
    #Do nothing.
    if k == 255:
        continue
    #Case ESQ - Exit program.
    elif k == 27:
        break
    #Case c or C - Change resolution.
    elif k == 99 or k == 67:
        if selectImage is not None:
            selectImage = changeResolution(selectImage, 64,64)
            cv.imshow("Selected image window", selectImage)
            print(selectImage.shape)
    #Case r or R - Reset image.
    elif k == 114 or k == 82:
        image = copyImage
    #Case t or T - Test function.
    elif k == 116 or k == 84:
        changeGrayScale(2)
    #Identify the key.
    else:
        print(k)