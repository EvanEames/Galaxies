print("=================\nEvan Eames - 9128249\nPHYS 60441 - Final Assignment Part 1\nDecember 21st 2014\n=================")

import numpy as np
import pyfits as pf
import math
import scipy
from scipy import optimize
from scipy.optimize import fmin
import matplotlib
from matplotlib import pyplot as plt

#Reading in file
fileName = raw_input("Please specify filename and path of desired fits file:\n")
a = pf.open(fileName)
a.info()
data = a[0].data
#For the Andromeda test file I was using there were some artifacts on the right edge I snipped off
data = data[0:7055,0:6541]
ySize,xSize = data.shape
#This finds the row and column of pixels that have the max average intensity, and assumes that this is the centre
x0=np.argmax(np.mean(data, axis=0))
y0=np.argmax(np.mean(data, axis=1))
print("Center located at:\nx0 = "+str(x0)+"\ny0 = "+str(y0))

#Get an idea of how high the intensity is at the center by sampling a tiny region
I_center = np.mean(data[y0-1:y0+1,x0-1:x0+1])

#Take 1D slices from the centre outwards in the 8 main cardinal directions
Eaxis = data[y0,x0:xSize]
Waxis = data[y0,0:x0+1][::-1]
Saxis = data[0:y0+1,x0][::-1]
Naxis = data[y0:ySize,x0]
SEaxis = np.diagonal(np.flipud(data[0:y0,x0:xSize]))
NEaxis = np.diagonal(data[y0:ySize,x0:xSize])
SWaxis = np.diagonal(data[0:y0,0:x0])[::-1]
NWaxis = np.diagonal(np.fliplr(data[y0:ySize,0:x0]))
axes = np.array([Eaxis,Waxis,Saxis,Naxis,NEaxis,SEaxis,NWaxis,SWaxis])
radii = []

#If you make this True it will show the cross-sections from the centre outwards in various directions
if False:
	plt.figure(1)
	plt.plot(NEaxis,'r',NWaxis,'b',SWaxis,'g',SEaxis,'c')
	plt.figure(2)
	plt.plot(Naxis,'r',Saxis,'b',Eaxis,'g',Waxis,'c')
	plt.show()

#This tries to guess where the galaxy ends in each of the 8 cardinal directions
#The reason for this is that it improves radius estimates for elliptic galaxies
for x in axes:
	#An attempt to chose a minimal intensity below which the galaxy is said to end.
	#Here this is said to be .5 sigma below the average. There is probably a better way of doing this (e.g. Gaussian fitting)
	Icutoff = np.mean(x)-0.5*np.std(x)
	#Any point that falls below Icutoff is 0, any above is 1
	intensityBinary = np.where(x > Icutoff,1,0)
	#As it is possible for pixels inside the galaxy to occasionally fall below Icutoff, it is necessary to define how long a string of zeros must be in order to decide the galaxy has ACTUALLY ended. I chose to make this one twentieth of the length of directional slice
	if len(x) > 60:
		segment = int(math.ceil(len(x)/20))
	else:
		#Or 3 if a directional slice is too short
		segment = 3
	for i in range (0,len(x)):
		#When a segment of the directional slice has more 0s than 1s, decide that halfway along this segment is where the galaxy ends
		if np.mean(intensityBinary[i:i+segment])<0.5:
			radii.append(i+math.ceil(segment/2))
			break
radii = np.asarray(radii)
#The diagonal directions need to be multiplied by sqrt(2) because we are counting the distance diagonally along 1x1 pixels
radii[4:8] *= math.sqrt(2)
#Set r0 to be the average radius along all 8 directions
r0 = np.mean(radii)
print("The average radius is " + str(r0) + " pixels.")


x = np.arange(-x0,xSize-x0,1.0)
y = np.arange(-y0,ySize-y0,1.0)
xx,yy = np.meshgrid(x,y)
r = np.hypot(xx,yy)
#pixels inside r0 are given values of 1, outside are given values of 0, then the max intensity inside r0 is called I0
rBinary = np.where(r < r0,1,0)
I0 = np.max(rBinary*data)
print("The maximum detected intensity is " + str(I0))

#Attempt a sersic profile and calculate goodness
def myfunc(n,data,r,r0,I0):
	sersic = I0*np.exp(-(r/r0)**(1/n))
	print("n="+str(n))
	goodness = np.mean((sersic - data)**2)
	print("goodness="+str(goodness))
	return(goodness)

#If runtime is too high, turn down tolerableError
tolerableError=100
scipy.optimize.fmin(myfunc,2,args = (data,r,r0,I0),xtol=tolerableError,ftol=tolerableError,maxfun=100,maxiter=100)
