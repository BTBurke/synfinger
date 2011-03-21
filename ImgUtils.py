#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

#class ImgUtils:
    
#    def __init__(self):
#        pass

def scaleImg(img):
    if np.max(img) > 255.0:
        if (np.max(img)-np.min(img))>0:
                img*=(255.0/(np.max(img)-np.min(img)))
                if np.min(img) < 0:
                    img+=abs(np.min(img))
                else:
                    img-=abs(np.min(img))
    img = np.rint(img)
    return img

def binarize(img):
    """
    Uses Otsu's Method to determine optimum threshold for binarization.
    Returns the binary image (grayscale colormap).
    """
    # Scale and cast to int if not already done 
    img = scaleImg(img)

    # Get histogram of values [0,255]            
    histImg, binEdges = np.histogram(img,range(0,257))
    
    # Convert arrays to floats
    histImg = np.asfarray(histImg)
    binEdges = np.asfarray(binEdges)

    varRatio = np.zeros((256,1),dtype='float')
    varRatio[0] = 10.0**10 # set high value because threshold of 0 shouldn't work
    totalSum = np.sum(histImg)
    
    for i in range(1,256):
        sumBelow = np.sum(histImg[:i])
        sumAbove = np.sum(histImg[i:])
        
        # Calculate weight, mean, variance below threshold (background)
        Wb = sumBelow / totalSum
        Mb = np.sum(np.multiply(binEdges[:i],histImg[:i])) / sumBelow
        Sb = np.sum(np.multiply((binEdges[:i]-Mb)**2,histImg[:i])) / sumBelow

         # Calculate weight, mean, variance above threshold (foreground)
        Wf = sumAbove / totalSum
        Mf = np.sum(np.multiply(binEdges[i:-1],histImg[i:])) / sumAbove
        Sf = np.sum(np.multiply((binEdges[i:-1]-Mf)**2,histImg[i:])) / sumAbove
        
        # Save inter-class weighted variance
        varRatio[i] = Wb*Sb + Wf*Sf

    # Do thresholding at location of minimum variance
    imgThresh = np.argmin(varRatio)
    img[np.where(img < imgThresh)] = 0
    img[np.where(img >= imgThresh)] = 255

    return img

#img=Image.open('otsu-test.jpg')
#img = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5]
#print np.shape(img)
#x = ImgUtils()
#imgThresh = x.binarize(img)
#plt.imshow(imgThresh, cmap=cm.gray);plt.show()

