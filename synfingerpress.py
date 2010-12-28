#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import scipy.signal as spsig
from scipy.ndimage import morphology
import math
import random
#from synfinger import SynFinger
from ImgUtils import ImgUtils
import uuid

class SynFingerPress(ImgUtils):
    
    def __init__(self):
        pass

    def blobNoise(self,img,level=None):
        """
        Produces random noise "blobs" where density is inversely proportional 
        to distance from the center of the image. Does post-processing filtering
        to make it look authentic.  Noise only applies to the "ink" regions
        """
        
        img = np.array(img)
        numRows, numCols = np.shape(img)
        noiseAdd = np.zeros((numRows,numCols), dtype='float')
        
        #bSize = 4
        rCen = np.ceil(numRows / 2)
        cCen = np.ceil(numCols / 2)
        
        # Set output SNR (High, Med, Low)
        if not level:
            level = random.choice(('high','medium','low'))
        if level == 'high':
            N = 100
        elif level == 'medium':
            N = 1500
        else:
            N = 3500

        for n in range(0,N):
            print n
            bSize = random.choice((2,4,6,8))
            r, c = (np.random.randint(0,numRows),np.random.randint(0,numCols))
            # Distance to edge along angle between random pt and center
            theta = math.atan2((r-rCen),(c-cCen))
            cosTh = math.cos(theta)
            sinTh = math.sin(theta)
            
            if (cosTh > 0.0) & (sinTh > 0.0):
                maxDist = min([float(rCen)/sinTh,float(cCen)/cosTh])
            elif (cosTh > 0.0):
                maxDist = cCen
            else:
                maxDist = rCen
            
            maxDist = math.floor(maxDist)
            
            distRatio = math.sqrt((r-rCen)**2+(c-cCen)**2)/maxDist

            # Put noise blob with probability proportional to distance
            # Noise spatial SNR inversely proportional to distance from center
            if np.random.rand(1) < distRatio:
                rL = max(0, r-bSize/2)
                rH = min(numRows, r+bSize/2)
                cL = max(0, c-bSize/2)
                cH = min(numCols, c+bSize/2)

                noiseAdd[rL:rH,cL:cH] = 255.0

        # Filter for better look, average then gaussian
        avgFilt = np.ones((8,8),dtype='float')/64
        gaussFilt = np.array([[0.011344, 0.083820, 0.011344],[0.083820, 0.619347, 0.083820],[0.011344, 0.083820, 0.011344]])

        noiseAdd = spsig.fftconvolve(noiseAdd, avgFilt, mode='same')
        noiseAdd = spsig.fftconvolve(noiseAdd, gaussFilt, mode='same')
        noiseAdd = spsig.fftconvolve(noiseAdd, gaussFilt, mode='same')

        noiseAdd = noiseAdd / np.max(noiseAdd) * 255.0

        # Add noise only to areas with fingerprint "ink"
        inkIndex = np.where(img < 255)
        valleyIndex = np.where(img == 255)
        img[inkIndex] += noiseAdd[inkIndex]
        #genNoise = 10.0 * np.random.randn(numRows,numCols) + 50.0
        #img[inkIndex] += genNoise[inkIndex]

        img = spsig.fftconvolve(img, avgFilt, mode='same')
        img = spsig.fftconvolve(img, gaussFilt, mode='same')

        img[valleyIndex] = 255
        img[inkIndex] = ImgUtils.scaleImg(self, img[inkIndex])
        print np.max(np.max(img))

        return img

    def makePrint(self, img, size=(400,400), rotate=True):
        """
        Converts to TIFF image and saves with UUID filename
        """
        img = np.array(img)
        size = np.shape(img)
        fname = uuid.uuid4()
        
        numRows, numCols = np.shape(img)
        im = Image.new("L",(numCols,numRows),255)
        for r in range(0,numRows):
            for c in range(0,numCols):
                #print img[r,c]
                im.im.putpixel((c,r), img[r,c])
        
        im.save(str(fname)+'.tif')
        
    def erodeDilate(self, img, method=None):
        """
        Use morphological operators to erode or dilate the ridge structure.
        Dilate uses a recursive call to first dilate then erode.  Dilation
        alone produces ridge structures that are too thick to look
        authentic. Recursive call introduces random spurious minutiae when
        some valley structures are bridged.
        """
        img = np.array(img)
        if not method:
            method = random.choice(('erode', 'dilate', 'none'))
        inkIndex = np.where(img < 250)
        imgBin = np.zeros(np.shape(img))
        imgBin[inkIndex] = 1
        
        strel = morphology.generate_binary_structure(2,2)
        if method == 'erode':
            imgBin = morphology.binary_erosion(imgBin, strel)
        elif method == 'dilate':
            imgBin = morphology.binary_dilation(imgBin, strel)
        else:
            return img

        inkIndex = np.where(imgBin == 1)
        returnImg = 255*np.ones(np.shape(img))
        returnImg[inkIndex] = 0
        
        # Recursive call to erode after dilation to give more authentic
        # appearance.  Erode after dilate introduces spurious minutiae
        # but does not make the ridge structure too thick
        if method == 'dilate':
            self.erodeDilate(returnImg, method='erode')
        
        return returnImg

img = Image.open('test.tif')
print np.shape(img)
x = SynFingerPress()
edImg = x.erodeDilate(img,'dilate')
#edImg2 = x.erodeDilate(edImg, 'erode')
blobX = x.blobNoise(edImg,'medium')
x.makePrint(blobX)
