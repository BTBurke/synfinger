#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as lines
from PIL import Image
import scipy as sp
import scipy.signal as spsig
import scipy.io as sio
import scipy.ndimage.filters as filt
import math
import random
from mpl_toolkits.mplot3d import axes3d
from ImgUtils import ImgUtils
#import scipy.stsci.convolve as convolve


class SynFinger(ImgUtils):
  """
  Generates a synthetic finger master image according to the SFINGE method. 
  Additional parameters added to simulate rolled prints.
  """
  
  def __init__(self):
    pass

  def genMask(self,a1,a2,b1,b2,c,d=0):
    """
    Generates a finger foreground mask with a center rectangle dxc (WxH) and
    major axes (b1,b2) (top,bottom), minor axes (a1,a2) (right,left)
    """

    # Force even numbers for a well-defined center point
    if c % 2 == 1: c=c+1
    if d & d % 2 == 1: d=d+1

    # Calculate limits of the mask and create zero array
    numRows=c+b1+b2
    numCols=d+a1+a2
    mask=np.zeros((numRows,numCols),dtype='int8')

    # Loop through each row starting at top of image
    # Calculate left and right limits and set mask to 1 inbetween
    for r in range(0,numRows):
      if r >= b1+c:
        # Ellipses defined by b2 major axis, a2/a1 minor (notation reversed)
        centerR=b1+c;
        centerC=a2+(d/2);
        leftLim=centerC-float(a2)*math.sqrt(1.0-(float(r-centerR)/float(b2))**2)-(d/2)
        rightLim=centerC+float(a1)*math.sqrt(1.0-(float(r-centerR)/float(b2))**2)+1+(d/2)
      elif r > b1:
        # In the middle rectangle, no ellipses required
        leftLim=0
        righLim=numCols
      else:
        # Ellipses defined by b1 major, a2/a1 minor axes (notation reversed)
        centerR=b1;
        centerC=a2+(d/2);
        leftLim=centerC-float(a2)*math.sqrt(1.0-(float(centerR-r)/float(b1))**2)-(d/2)
        rightLim=centerC+float(a1)*math.sqrt(1.0-(float(centerR-r)/float(b1))**2)+1+(d/2)
    
      # Check for out of bounds due to float math, truncate to int
      if leftLim < 0: 
        leftLim =0 
      else:
        leftLim=int(leftLim)
      if rightLim > numCols: 
        rightLim=numCols 
      else: 
        rightLim=int(rightLim)
      
      # Set mask to 1 in areas where fingerprint will be generated
      mask[r,leftLim:rightLim]=1

    return mask

  def makeSingularPts(self, mask, type=None):
        """
        Creates loop and delta points based on the Henry class (randomized placement)
        """
        numRows,numCols = np.shape(mask)
        if not type:
            type = random.choice(('Arch','Left Loop','Right Loop','Tented Arch','Whorl'))
        if type == 'Arch':
            # Put a loop below mask area FIXME: This doesn't work (maybe some kind of gaussian?)
            ls=[(random.randint(numRows,2*numRows),random.randint(.2*numCols,.8*numCols))]
            ds=[]
        elif type == 'Left Loop':
            # Put one loop in top half of fingeprint, center +/- 10% 
            ls=[(random.randint(int(.4*numRows),int(.6*numRows)),random.randint(int(.4*numCols),int(.6*numCols)))]
            # Find maximum distance to edge of fingerprint, put 1 delta at random
            # angle in correct quadrant
            dsOffDist=random.uniform(.2,.7)*math.sqrt((numRows-ls[0][0])**2+(numCols-ls[0][1])**2)
            dsOffAngle=random.uniform(math.pi/8.0,3.0*math.pi/8.0)
            ds=[(ls[0][0]+int(dsOffDist*math.sin(dsOffAngle)),ls[0][1]+int(dsOffDist*math.cos(dsOffAngle)))]
        elif type == 'Right Loop':
            # Put one loop in top half of fingeprint, center +/- 20% 
            ls=[(random.randint(int(.4*numRows),int(.6*numRows)),random.randint(int(.4*numCols),int(.6*numCols)))]
            # Find  offset distance (40-90% to edge of fingerprint), put 1 delta at random
            # angle in correct quadrant
            dsOffDist=random.uniform(.2,.7)*math.sqrt((numRows-ls[0][0])**2+(ls[0][1])**2)
            dsOffAngle=random.uniform(math.pi/8.0,3.0*math.pi/8.0)+math.pi/2.0
            ds=[(ls[0][0]+int(dsOffDist*math.sin(dsOffAngle)),ls[0][1]+int(dsOffDist*math.cos(dsOffAngle)))]
        elif type == 'Tented Arch':
            pass
        elif type == 'Whorl':
            pass
        else:
            assert false, 'Invalid Fingerprint Type'
        
        return(ls, ds)

  def makeOrientationMap(self,ls,ds,mask):
        """
        Generates an orientation map based on the Henry class of fingerprint
        """
        numRows, numCols = np.shape(mask)
        orientMap = np.zeros((numRows,numCols),dtype='float')
        
        # Calculate the Sherlock-Munro Model with Vizcaya-Gerhardt Correction TODO: V-G Correction
        # Signs reversed due to way rows are indexed in Python
        L=8
        g_alpha_i=[-1*math.pi+2*math.pi*i/L for i in range(0,L)]
        for r in range(0,numRows):
            for c in range(0,numCols):
                Z_ds=np.sum([math.atan2((r-ds[i][0]),(c-ds[i][1])) for i in range(0,len(ds))])
                Z_ls=np.sum([math.atan2((r-ls[i][0]),(c-ls[i][1])) for i in range(0,len(ls))])
                orientMap[r,c]=0.5*(Z_ls-Z_ds)
        return orientMap
        
  def gaborFilter(self,orientMap):
        """
        Applies Gabor filters to generate a ridge structure based on local orientation
        """
        numRows, numCols = np.shape(orientMap)
        masterImage=np.zeros((numRows,numCols),dtype='float')
        
        # Create a spatially varying frequency response
        # Uses a tukey window to lower freq above/below singular pts
        nr = np.linspace(0,1,numRows)
        tukey = np.ones(nr.shape,dtype='float')
        alpha = 0.7
        first = nr < alpha/2.0
        tukey[first] = 0.5 * (1 + np.cos(2*np.pi/alpha * (nr[first] - alpha/2.0) ))
        third = nr >= 1-alpha/2.0
        tukey[third] = 0.5 * (1 + np.cos(2*np.pi/alpha * (nr[third] - 1 + alpha/2.0)))
        tukey = 3.0*(1.0-tukey)
        spatialFreq = 7.5 * np.ones((numRows,numCols),dtype='float')
        for r in range(0,numRows):
            spatialFreq[r]=spatialFreq[r]+tukey[r]


        for n in range(1,1000):
            print n
            # Seed with N initial points, keeping them inside filter overlap boundary 
            # so no edge effects during inital seeding
            filtSize=16
            N=8
            outBound=filtSize/2+1
            seedPointsR=np.random.randint(outBound,numRows-outBound,N)
            seedPointsC=np.random.randint(outBound,numCols-outBound,N)
    
            # During first two iterations, put a dirac to stimulate filter
            # response 
            if n<2:
                for i in range(0,N):
                    masterImage[seedPointsR[i],seedPointsC[i]]=1.0
            

            # Do Monte Carlo sampling of N points, applying Gabor filter at
            # at each point
            for i in range(0,N):
                #f=1/6.0 #TODO: implement spatially varying frequency
                #sig = 4.0 #TODO: calculate sigma based on local freq
                th = orientMap[seedPointsR[i],seedPointsC[i]]-math.pi/2.0

                r = seedPointsR[i]
                c = seedPointsC[i]
                
                f = 1.0 / spatialFreq[r,c]
                sig = -1.0 * (3.0/(2.0*f))**2 / math.log(10.0**(-3)) / 2.0

                filtCoef=[[math.exp(-1.0*(float(x)**2+float(y)**2)/(2.0*sig))*math.cos(2.0*math.pi*f*(float(x)*math.cos(th)+float(y)*math.sin(th))) for x in range(-filtSize/2,filtSize/2)] for y in range(-filtSize/2,filtSize/2)]
                filtCoef=np.array(filtCoef)
                           
                fs2=filtSize/2
                appArea=masterImage[r-fs2:r+fs2,c-fs2:c+fs2].copy()
                testVar=spsig.fftconvolve(appArea,filtCoef,mode='same')
                
                # Test for malformed response and replace
                if np.isnan(testVar).any():
                    testVar[np.where(np.isnan(testVar))]=0.0
                if np.isinf(testVar).any():
                    testVar[np.where(np.isinf(testVar))]=0.0
                
                # Scale each time to adjust contrast    
                if (np.max(testVar)-np.min(testVar))>0:
                    testVar*=(255.0/(np.max(testVar)-np.min(testVar)))
                    testVar+=abs(np.min(testVar))
                    
                masterImage[r-fs2:r+fs2,c-fs2:c+fs2]=testVar
        
        # After all seeding complete, do entire image with 50% window overlap
        for r in range(fs2,numRows-fs2,fs2):
            for c in range(fs2,numCols-fs2,fs2):
                th = orientMap[r,c]-math.pi/2.0

                f = 1.0 / spatialFreq[r,c]
                sig = -1.0 * (3.0/(2.0*f))**2 / math.log(10.0**(-3)) / 2.0


                filtCoef=[[math.exp(-1.0*(float(x)**2+float(y)**2)/(2.0*sig))*math.cos(2.0*math.pi*f*(float(x)*math.cos(th)+float(y)*math.sin(th))) for x in range(-filtSize/2,filtSize/2)] for y in range(-filtSize/2,filtSize/2)]
                    
                filtCoef=np.array(filtCoef)
                           
                fs2=filtSize/2
                appArea=masterImage[r-fs2:r+fs2,c-fs2:c+fs2].copy()
                testVar=spsig.fftconvolve(appArea,filtCoef,mode='same')
                    
                if np.isnan(testVar).any():
                    testVar[np.where(np.isnan(testVar))]=0.0
                if np.isinf(testVar).any():
                    testVar[np.where(np.isinf(testVar))]=0.0
                    
                if (np.max(testVar)-np.min(testVar))>0:
                    testVar*=(255.0/(np.max(testVar)-np.min(testVar)))
                    testVar+=abs(np.min(testVar))

                masterImage[r-fs2:r+fs2,c-fs2:c+fs2]=testVar

        # Moving average filter to get rid of windowing effects            
        masterImage = spsig.fftconvolve(masterImage,np.ones((2,2),dtype='float'),mode='same')
        masterImage = ImgUtils.scaleImg(self, masterImage)
        return masterImage

  def applyMask(self,thresImage,mask):
        """
        Applies the finger shape mask to the ridge structure and returns print
        """
        maskImage = np.multiply(mask,threshImage)
        maskImage[np.where(mask == 0)] = 255
        return maskImage


finger = SynFinger()
mask = finger.genMask(100,120,120,110,70,d=0)
ls,ds = finger.makeSingularPts(mask, 'Right Loop')
orientMap = finger.makeOrientationMap(ls,ds,mask)
masterImage = finger.gaborFilter(orientMap)
threshImage = finger.binarize(masterImage)
maskImage = finger.applyMask(threshImage,mask)
print np.shape(maskImage)

numRows, numCols = np.shape(maskImage)
im = Image.new("L",(numCols,numRows),255)
for r in range(0,numRows):
    for c in range(0,numCols):
        im.im.putpixel((c,r),maskImage[r,c])

im.save('test.tif')

#print np.max(masterImage),np.min(masterImage)
#plt.figure()
plt.imshow(maskImage, cmap=cm.gray)

plt.plot([r[1] for r in ls],[y[0] for y in ls],'o')
plt.plot([r[1] for r in ds],[y[0] for y in ds],'^')
#numRows, numCols = np.shape(orientMap)
#for r in range(0,numRows,3):
#    for c in range(0,numCols,3):
#        plt.plot([c,c+2.0*math.cos(orientMap[r][c])],[r,r+2.0*math.sin(orientMap[r][c])],'g-')
        


plt.show()

