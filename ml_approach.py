import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
K.set_image_data_format('channels_first')



def show_images(image_path_original,image_path_check):
    
    imageA=mpimg.imread(image_path_original)
    imageB=mpimg.imread(image_path_check)
    fig = plt.figure("Comparison")
 
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
 
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
 
    # show the images
    plt.show()

def verify(image_path_original, image_path_check, model): 
    
    encoding_original = img_to_encoding(image_path_original,model)
    encoding_check = img_to_encoding(image_path_check,model)
  
    dist = np.linalg.norm(encoding_original-encoding_check)
    # setup the figure
    
    show_images(image_path_original,image_path_check)
    
    print ("Distance is " , dist)
    if dist < 0.7:
        print("It's similar")
                
    else:
        print("It's not similar")
    return dist

def verify_ml_approach(image_path_original,image_path_check):

    dist = verify(image_path_original, image_path_check, FRmodel)
    
    return dist
    