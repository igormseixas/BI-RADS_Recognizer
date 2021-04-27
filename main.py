import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

from tkinter import *
from tkinter import filedialog

from PIL import Image
from PIL import ImageTk


# Global control variables.
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

# Configure Main Menu.
def configure_menu():
    #File Menu.
    file_menu = Menu(menu, tearoff=0) # Tearoff=0 makes the menu cleaner.
    menu.add_cascade(label="File",menu=file_menu) # Add menu cascade for file.
    file_menu.add_command(label ='New File', command = None)
    file_menu.add_command(label="Open...",command=open_image) # Open option inside file menu.
    file_menu.add_command(label ='Save', command = None)
    file_menu.add_separator()
    file_menu.add_command(label ='Clear', command=lambda: reset_area(image_area=True))
    file_menu.add_separator()
    file_menu.add_command(label ='Exit', command = root.destroy)

    #Region Menu.
    region_menu = Menu(menu,tearoff=0)
    menu.add_cascade(label="Region",menu=region_menu) # Add menu cascade for region.
    
    region_sampling = Menu(region_menu,tearoff=0) # Submenu sampling.
    region_menu.add_cascade(label="Sampling Rate", menu=region_sampling) # Add menu cascade for sampling inside region.
    region_sampling.add_command(label="32x32",command=lambda: changeResolution(newWidth=32, newHeight=32)) # Sampling 32x32 region option inside region menu.
    region_sampling.add_command(label="64x64",command=lambda: changeResolution(newWidth=64, newHeight=64)) # Sampling 64x64 region option inside region menu.

    region_quantization = Menu(region_menu,tearoff=0) # Submenu quantization.  
    region_menu.add_cascade(label="Quantization Level", menu=region_quantization) # Add menu cascade for quantization inside region.
    region_quantization.add_command(label="16",command=lambda: changeGrayScale(newGrayScale=16)) # Quantizating 16 level of grays region option inside region menu.
    region_quantization.add_command(label="32",command=lambda: changeGrayScale(newGrayScale=32)) # Quantizating 32 level of grays region option inside region menu.
    region_quantization.add_command(label="256",command=lambda: changeGrayScale(newGrayScale=256)) # Quantizating 256 level of grays region option inside region menu.
    
    region_menu.add_command(label="Histogram Equalization",command=equalize) # Image histogram equalization option inside image menu.

    region_menu.add_separator()
    region_menu.add_command(label="Reset",command=lambda: reset_area(image_area=False)) # Reset region option inside region menu.

# Convert image from opencv to imagetk format to show at Tk. 
# Return new image to show.
def convert_image(image):
    imageTk = cv.cvtColor(image, cv.COLOR_BGR2RGB) #ImageTk format is RGB and opencv is BGR
    imageTk = Image.fromarray(imageTk)
    imageTk = ImageTk.PhotoImage(imageTk)
    return imageTk

# Apply ImageTk format in the label to show at Tk.
def apply_image(image,label):
    label.configure(image=image)
    label.image = image

# Redefine image or region as it was before.
def reset_area(image_area): # Image_area True for image; False to reset region.
    global image
    global selectedRegion
    global selectedRegion_is_gray
    global zoom_in
    if (image_label is not None) and image_area:
        image = copyImage.copy() # Reset image.
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
        zoom_in = False # Removes zoom_in if True.

        region_window.destroy() # Close the select region window.
    elif (region_window is not None) and (region_window.winfo_exists()):
        selectedRegion = copySelectedRegion.copy() # Reset region.
        selectedRegion_is_gray = False # Removes selectedRegion_is_gray if True.
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)

# Open a path to choose image, save its original format in opencv and show result at ImageTk.
def open_image():
    global image_label
    global copyImage
    global image
    global originalWidth
    global originalHeight

    path = filedialog.askopenfilename() # Gets path.
    if len(path) > 0:
        image = cv.imread(path, cv.IMREAD_COLOR) # Read as opencv.
        copyImage = image.copy() # Save original image.
        originalWidth = image.shape[1]
        originalHeight = image.shape[0]

        imageTk = convert_image(image)
        # If there is no image yet, create a label at Tk to show it.
        if image_label is None:
            image_label = Label(image=imageTk)
            image_label.image = imageTk
            image_label.bind("<Double-Button-1>", doubleclick_event) # Event with double left click at the image.
            image_label.bind("<MouseWheel>", mousewheel_event) # Event with mousewheel click at the image.
            image_label.pack() # Create and show label/image at Tk.
        else:
            apply_image(imageTk,image_label) #if label already exists, just apply new image to it
            if (region_window is not None) and (region_window.winfo_exists()):
                region_window.destroy()

# Create blue square as selectedRegion with 128x128, makes a copy of it, create a new window for the region and show it.
def doubleclick_event(event):
    global selectedRegion
    global selectedRegion_is_gray
    global copySelectedRegion
    global region_window
    global region_label
    global image
    
    if not zoom_in: #avoiding making a square while zooming in
        # Clears image before making a new sub-rect.
        image = copyImage.copy()
        # Reset the gray scale situation.
        selectedRegion_is_gray = False
        # Crop the sub-rect from the image
        region_size = 128 #Region size may change to 128, 64 and 32.
        # Making the region size as parameter, checks if it will be out of bounds first.
        # First is width.
        if (event.x-(region_size//2)) < 0: 
            region_initial_width = 0
            region_final_width = region_size-1
        elif (event.x+(region_size//2)) > originalWidth:
            region_initial_width = originalWidth - region_size
            region_final_width = originalWidth - 1
        else: # If it has space from both sides.
            region_initial_width = event.x-(region_size//2)
            region_final_width = (event.x+(region_size//2)) - 1
        # Same for height.
        if (event.y-(region_size//2)) < 0:
            region_initial_height = 0
            region_final_height = region_size-1
        elif (event.y+(region_size//2)) > originalHeight:
            region_initial_height = originalHeight - region_size
            region_final_height = originalHeight - 1
        else:
            region_initial_height = event.y-(region_size//2)
            region_final_height = (event.y+(region_size//2)) - 1
        # Crop region within range.
        overlay = image[region_initial_height:region_final_height, region_initial_width:region_final_width]
        # Copy information to a selected image.
        selectedRegion = overlay.copy()
        # Define a blue rectangle in the same shape as previously selected.
        blue_rect = np.full(overlay.shape, (255,0,0), dtype=np.uint8) #Build rectangle and set the blue color (255,0,0)
        # Add the rectangle to the selected and previously cut area of an image. Scale the transparency.
        #cv.addWeighted(overlay, 0.5, blue_rect, 0.5, 1.0)
        transparency=0.7 #Greater the value, greater the transparency is.
        gamma=10.0 # Gamma of the selected area, more gamma more white will me added.
        select = cv.addWeighted(overlay, transparency, blue_rect, 1-transparency, gamma)
        image[region_initial_height:region_final_height, region_initial_width:region_final_width] = select
        # Convert and show at Tk.
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
        
        copySelectedRegion = selectedRegion.copy() # Save original region.
        #Show area of interest in another window.
        selectedRegionTk = convert_image(selectedRegion)

        # Get root window position.
        x = root.winfo_width() + root.winfo_x()
        y = root.winfo_y()

        # If there is no window for it yet, creates one; else, just updates it.
        if region_window is None:
            region_window = Toplevel()
            region_window.title("Região")

            region_label = Label(region_window,image=selectedRegionTk) # Creating new label for region inside new window.
            region_label.image = selectedRegionTk
            region_label.pack(side = TOP, pady = 10) # Show region.
            #250x180 is the box of the window. x+100 is the padding of the width, and y + 0 is the padding of the height.
            region_window.geometry("%dx%d+%d+%d" % (250, 180, x + 100, y + 0)) 
            region_window.mainloop() # Starts new window.
        else:
            # Makes sure if the window is still open; if not, recreate it.
            if region_window.winfo_exists():
                # Reset the transformations of the image.
                apply_image(selectedRegionTk,region_label)
            else:
                region_window = Toplevel()
                region_window.title("Região")

                region_label = Label(region_window,image=selectedRegionTk) # Creating new label for region inside new window.
                region_label.image = selectedRegionTk
                region_label.pack(side = TOP, pady = 10) # Show region.
                #250x180 is the box of the window. x+100 is the padding of the width, and y + 0 is the padding of the height.
                region_window.geometry("%dx%d+%d+%d" % (250, 180, x + 100, y + 0)) 
                region_window.mainloop() # Starts new window.

# Zoom in and out scrolling mouse wheel centering around pointer; 
# Updates image with zoom in and show at Tk; if with zoom in, zoom out restore original image.
def mousewheel_event(event):
    global image
    global zoom_in
    # Sign of the delta shows direction of mousewheel.
    if event.delta > 0:
        zoom_in = True
        # Scroll up
        image = cv.resize(image,None,fx=1.1, fy=1.1, interpolation = cv.INTER_AREA)

        # New position of mouse pointer after resize.
        new_width = int(event.x * 1.1)
        new_height = int(event.y * 1.1)

        # Making the original image resolution as parameter, checks if it will be out of bounds first.
        # First is width.
        if (new_width - originalWidth//2) < 0: 
            initial_width = 0
            final_width = originalWidth-1
        elif (new_width + originalWidth//2) > image.shape[1]:
            initial_width = image.shape[1] - originalWidth
            final_width = image.shape[1] - 1
        else: # If it has space from both sides.
            initial_width = new_width - originalWidth//2
            final_width = (new_width + originalWidth//2) - 1
        # Same for height.
        if (new_height - originalHeight//2) < 0:
            initial_height = 0
            final_height = originalHeight-1
        elif (new_height + originalHeight//2) > image.shape[0]:
            initial_height = image.shape[0] - originalHeight
            final_height = image.shape[0] - 1
        else:
            initial_height = new_height - originalHeight//2
            final_height = (new_height + originalHeight//2) - 1
        # Update image within range.
        image = image[initial_height:final_height, initial_width:final_width]
        # Convert and show it.
        imageTk = convert_image(image)
        apply_image(imageTk,image_label)
    else:
        # Scroll down
        if zoom_in: # If it had zoommed in, reset image to its original state.
            reset_area(True) # Reset image

# Function to change level of gray in the region and updates it.
def changeGrayScale(newGrayScale):
    # Global copySelectedRegion
    global selectedRegion
    global selectedRegion_is_gray # To know when region is in gray scale.

    if (region_window is not None) and (region_window.winfo_exists()): # Makes sure the window with region exists.
        #Convert image to Gray if not already.
        if not selectedRegion_is_gray: # Avoid error if region is already in gray scale.
            selectedRegion = cv.cvtColor(selectedRegion, cv.COLOR_BGR2GRAY)
            selectedRegion_is_gray = True

        newGrayScale = newGrayScale-1 # 16 levels of gray is 0 to 15.
        for i in range(0,selectedRegion.shape[0]):
            for j in range(0,selectedRegion.shape[1]):
                #To quantize each channel into N levels (assuming the input is in the range [0,255]. 
                # You can also use floor, which is more suitable in some cases. 
                # This leads to N^3 different colors. 
                # For example with N=8 you get 512 unique RGB colors.
                # round(img*(N/255))*(255/N)
                selectedRegion[i,j] = round(selectedRegion[i,j]*(newGrayScale/255))*(255/newGrayScale) # Uniform quantization.

        # Convert and show region.
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)

# Function to change resolution of the region and updates it.
def changeResolution(newWidth, newHeight):
    global selectedRegion

    if (region_window is not None) and (region_window.winfo_exists()): # Makes sure the window with region exists.
        #Dimenstion we wanted to be change.
        dimension=(newWidth,newHeight)   
        #First set the resolution to the image.
        selectedRegion = cv.resize(selectedRegion,dimension, interpolation = cv.INTER_AREA)
        #Return the image to your original size with the new resolution.
        selectedRegion = cv.resize(selectedRegion, (128,128), interpolation = cv.INTER_AREA)

        # Convert and show region.
        selectedRegionTk = convert_image(selectedRegion)
        apply_image(selectedRegionTk,region_label)        

# Function to equalize image.
def equalize():
    global selectedRegion

    if (region_window is not None) and (region_window.winfo_exists() and selectedRegion_is_gray): # Makes sure the window with region exists.
        equalize_img = cv.equalizeHist(selectedRegion)

        # Convert and show region.
        selectedRegionTk = convert_image(equalize_img)
        apply_image(selectedRegionTk,region_label)  

root = Tk() # Main instance of Tk application.
root.title("BI-RADS Recognizer")
root.geometry("+500+75") # Move the main window to 400, 400 on the screen.

# Create Main Menu.
menu = Menu(root)
root.configure(menu=menu)
configure_menu() #Procedure to create our menu.

# Runs final application.
root.mainloop()