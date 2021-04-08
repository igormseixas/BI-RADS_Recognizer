import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys


#Read the image and verify if is it loaded correctly.
image = cv.imread("d_left_mlo (44).png")
if image is None:
    sys.exit("Could not read the image.")

#Get original image width and height for futher use.
originalWidth = image.shape[1]
originalHeight = image.shape[0]

#Mouse boolean signs.
#mousewheelUp = False
#mousewheelDown = False

#Mouse functions.
def mouse_events(event, x, y, flags, param):
    #print(event)
    #print(flags)
    global image
    
    if event == cv.EVENT_LBUTTONDBLCLK:
        #Crop the sub-rect from the image
        overlay = image[y:y+128, x:x+128]
        blue_rect = np.full(overlay.shape, (255,0,0), dtype=np.uint8) #Build rectangle and set the blue color (255,0,0)

        #Add the rectangle to the selected and previously cut area of an image. Scale the transparency.
        cv.addWeighted(overlay, 0.5, blue_rect, 0.5, 1.0)
        transparency=0.7 #Greater the value, greater the transparency is.
        gamma=10.0 #Gamma of the selected area, more gamma more white will me added.
        select = cv.addWeighted(overlay, transparency, blue_rect, 1-transparency, gamma)
        image[y:y+128, x:x+128] = select
    
    if event == cv.EVENT_MOUSEWHEEL:
        #zoom_image = image
        #Sign of the flag shows direction of mousewheel.
        if flags > 0:
            #scroll up
            #cv.resize(zoom_image,None,fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC)
            #image = zoom_image
            image = cv.resize(image,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
        else:
            print(originalWidth < image.shape[1])
            print(originalHeight < image.shape[0])
            #scroll down
            if originalWidth < image.shape[1] and originalHeight < image.shape[0]:
                image = cv.resize(image,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
        
        print(image.shape)

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
    
    if k == 27:
        break
    
    #if  mousewheelUp:
    #    print("AQUI")
    #elif mousewheelDown:
    #    print("Outro AQUI")