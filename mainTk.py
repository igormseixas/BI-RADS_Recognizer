import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# configure main menu
def configure_menu():
    #file menu
    file_menu = Menu(menu, tearoff=0) #tearoff=0 makes the menu cleaner
    menu.add_cascade(label="File",menu=file_menu) #add menu cascade for file
    file_menu.add_command(label="Open..",command=open_image) #open option inside file menu

    #image menu
    image_menu = Menu(menu,tearoff=0)
    menu.add_cascade(label="Image",menu=image_menu) #add menu cascade for image
    image_menu.add_command(label="Reset",command=lambda: reset_area(image_area=True)) #reset image option inside image menu
    image_menu.add_command(label="Histogram Equalization",command=None) #image histogram equalization option inside image menu

    #region menu
    region_menu = Menu(menu,tearoff=0)
    menu.add_cascade(label="Region",menu=region_menu) #add menu cascade for region
    region_menu.add_command(label="Reset",command=lambda: reset_area(image_area=False)) #reset region option inside region menu
    region_sampling = Menu(region_menu,tearoff=0) #submenu sampling
    region_menu.add_cascade(label="Sampling Rate", menu=region_sampling) #add menu cascade for sampling inside region
    region_sampling.add_command(label="32x32",command=lambda: changeResolution(newWidth=32, newHeight=32)) #sampling 32x32 region option inside region menu
    region_sampling.add_command(label="64x64",command=lambda: changeResolution(newWidth=64, newHeight=64)) #sampling 64x64 region option inside region menu
    region_quantization = Menu(region_menu,tearoff=0) #submenu quantization
    region_menu.add_cascade(label="Quantization Level", menu=region_quantization) #add menu cascade for quantization inside region
    region_quantization.add_command(label="16",command=lambda: changeGrayScale(newGrayScale=16)) #quantizating 16 level of grays region option inside region menu
    region_quantization.add_command(label="32",command=lambda: changeGrayScale(newGrayScale=32)) #quantizating 32 level of grays region option inside region menu
    region_quantization.add_command(label="256",command=lambda: changeGrayScale(newGrayScale=256)) #quantizating 256 level of grays region option inside region menu

#convert image from opencv to imagetk format to show at Tk. return new image to show
def convert_image(image):
    imageTk = cv.cvtColor(image, cv.COLOR_BGR2RGB) #ImageTk format is RGB and opencv is BGR
    imageTk = Image.fromarray(imageTk)
    imageTk = ImageTk.PhotoImage(imageTk)
    return imageTk

#apply ImageTk format in the label to show at Tk
def apply_image(image,label):
    label.configure(image=image)
    label.image = image

#redefine image or region as it was before
def reset_area(image_area): #image_area True for image; False to reset region
    global image
    global selectedRegion
    global selectedRegion_is_gray
    global zoom_in
    if (image_label is not None) and image_area:
        image = copyImage.copy() #reset image
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
        zoom_in = False #removes zoom_in if True
    elif (region_window is not None) and (region_window.winfo_exists()):
        selectedRegion = copySelectedRegion.copy() #reset region
        selectedRegion_is_gray = False #removes selectedRegion_is_gray if True
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)

#open a path to choose image, save its original format in opencv and show result at ImageTk
def open_image():
    global image_label
    global copyImage
    global image
    global originalWidth
    global originalHeight
    path = filedialog.askopenfilename() #gets path
    if len(path) > 0:
        image = cv.imread(path, cv.IMREAD_COLOR) #read as opencv
        copyImage = image.copy() #save original image
        originalWidth = image.shape[1]
        originalHeight = image.shape[0]
        #print(originalHeight, originalWidth)
        #print(image.shape[0],image.shape[1])
        imageTk = convert_image(image)
        # if there is no image yet, create a label at Tk to show it
        if image_label is None:
            image_label = Label(image=imageTk)
            image_label.image = imageTk
            image_label.bind("<Double-Button-1>", doubleclick_event) #event with double left click at the image
            image_label.bind("<MouseWheel>", mousewheel_event) #event with mousewheel click at the image
            image_label.pack() #create and show label/image at Tk
        else:
            apply_image(imageTk,image_label) #if label already exists, just apply new image to it

#create blue square as selectedRegion with 128x128, makes a copy of it, create a new window for the region and show it
def doubleclick_event(event):
    global selectedRegion
    global copySelectedRegion
    global region_window
    global region_label
    global image
    
    if not zoom_in: #avoiding making a square while zooming in
        #Clears image before making a new sub-rect.
        image = copyImage.copy()
        #Crop the sub-rect from the image
        region_size = 128 #Region size may change to 128, 64 and 32.
        overlay = image[event.y-(region_size//2):event.y+(region_size//2), event.x-(region_size//2):event.x+(region_size//2)]
        #Copy information to a selected image.
        selectedRegion = overlay.copy()
        #Define a blue rectangle in the same shape as previously selected.
        blue_rect = np.full(overlay.shape, (255,0,0), dtype=np.uint8) #Build rectangle and set the blue color (255,0,0)
        #Add the rectangle to the selected and previously cut area of an image. Scale the transparency.
        #cv.addWeighted(overlay, 0.5, blue_rect, 0.5, 1.0)
        transparency=0.7 #Greater the value, greater the transparency is.
        gamma=10.0 #Gamma of the selected area, more gamma more white will me added.
        select = cv.addWeighted(overlay, transparency, blue_rect, 1-transparency, gamma)
        image[event.y-(region_size//2):event.y+(region_size//2), event.x-(region_size//2):event.x+(region_size//2)] = select
        #convert and show at Tk
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
        
        copySelectedRegion = selectedRegion.copy() #save original region
        #Show area of interest in another window.
        selectedRegionTk = convert_image(selectedRegion)
        #print(region_window.state())
        #if there is no window for it yet, creates one; else, just updates it
        if region_window is None:
            #print(region_window.winfo_exists())
            #region_window.destroy()
            region_window = Toplevel()
            #region_window.winfo_exists()
            region_window.title("Região")
            region_label = Label(region_window,image=selectedRegionTk) #creating new label for region inside new window
            region_label.image = selectedRegionTk
            region_label.pack() #show region
            region_window.mainloop() #starts new window
        else:
            #makes sure if the window is still open; if not, recreate it
            if region_window.winfo_exists():
                apply_image(selectedRegionTk,region_label)
            else:
                region_window = Toplevel()
                region_window.title("Região")
                region_label = Label(region_window,image=selectedRegionTk)
                region_label.image = selectedRegionTk
                region_label.pack()
                region_window.mainloop()

#zoom in and out scrolling mouse wheel centering around pointer; 
# updates image with zoom in and show at Tk; if with zoom in, zoom out restore original image
def mousewheel_event(event):
    global image
    global zoom_in
    #Sign of the delta shows direction of mousewheel.
    if event.delta > 0:
        zoom_in = True
        #scroll up
        image = cv.resize(image,None,fx=1.1, fy=1.1, interpolation = cv.INTER_AREA)

        # new position of mouse pointer after resize
        new_x = int(event.x * 1.1) #width
        new_y = int(event.y * 1.1) #height
        #print(event.x,event.y)
        #print(new_x,new_y)
        #print(image.shape[0],image.shape[1])
        #making the original image resolution as parameter, checks if it will be out of bounds first
        #first is width
        if (new_x - originalWidth//2) < 0: 
            initial_width = 0
            final_width = originalWidth-1
        elif (new_x + originalWidth//2) > image.shape[1]:
            initial_width = image.shape[1] - originalWidth
            final_width = image.shape[1] - 1
        else: #if it has space from both sides
            initial_width = new_x - originalWidth//2
            final_width = (new_x + originalWidth//2) - 1
        #samething for height
        if (new_y - originalHeight//2) < 0:
            initial_height = 0
            final_height = originalHeight-1
        elif (new_y + originalHeight//2) > image.shape[0]:
            initial_height = image.shape[0] - originalHeight
            final_height = image.shape[0] - 1
        else:
            initial_height = new_y - originalHeight//2
            final_height = (new_y + originalHeight//2) - 1
        #print(initial_height,final_height,initial_width,final_width)
        #update image within range
        image = image[initial_height:final_height, initial_width:final_width]
        #convert and show it
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
    else:
        #print(originalWidth < image.shape[1])
        #print(originalHeight < image.shape[0])
        #scroll down
        #if originalWidth < image.shape[1] and originalHeight < image.shape[0]:
        if zoom_in: #if it had zoommed in, reset image to its original state
            #image = cv.resize(image,None,fx=0.9091, fy=0.9091, interpolation = cv.INTER_AREA)
            reset_area(True) #reset image

#change level of gray in the region and updates it
def changeGrayScale(newGrayScale):
    #global copySelectedRegion
    global selectedRegion
    global selectedRegion_is_gray #to know when region is in gray scale
    #print(newGrayScale)
    if (region_window is not None) and (region_window.winfo_exists()): #makes sure the window with region exists
        #Convert image to Gray if not already.
        if not selectedRegion_is_gray: #avoid error if region is already in gray scale
            selectedRegion = cv.cvtColor(selectedRegion, cv.COLOR_BGR2GRAY)
            selectedRegion_is_gray = True

        newGrayScale = newGrayScale-1 # 16 levels of gray is 0 to 15
        for i in range(0,selectedRegion.shape[0]):
            for j in range(0,selectedRegion.shape[1]):
                #To quantize each channel into N levels (assuming the input is in the range [0,255]. 
                # You can also use floor, which is more suitable in some cases. 
                # This leads to N^3 different colors. 
                # For example with N=8 you get 512 unique RGB colors.
                #round(img*(N/255))*(255/N)
                selectedRegion[i,j] = round(selectedRegion[i,j]*(newGrayScale/255))*(255/newGrayScale) #Uniform quantization.

        #convert and show region
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)

#change resolution of the region and updates it
def changeResolution(newWidth, newHeight):
    #global copySelectedRegion
    global selectedRegion
    #print(newWidth,newHeight)
    if (region_window is not None) and (region_window.winfo_exists()): #makes sure the window with region exists
        #Dimenstion we wanted to be change.
        dimension=(newWidth,newHeight)   
        #First set the resolution to the image.
        selectedRegion = cv.resize(selectedRegion,dimension, interpolation = cv.INTER_AREA)
        #Return the image to your original size with the new resolution.
        selectedRegion = cv.resize(selectedRegion, (128,128), interpolation = cv.INTER_AREA)

        #convert and show region
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)        


root = Tk() #main instance of Tk application
root.title("Projeto PI")

#global control variables
image_label = None
region_window = None
region_label = None
image = None
originalWidth = None
originalHeight = None
copyImage = None
selectedRegion = None
selectedRegion_is_gray = False
copySelectedRegion = None
zoom_in = False

# creates main menu
menu = Menu(root)
root.configure(menu=menu)
configure_menu() #procedure to create our menu

# runs final application
root.mainloop()