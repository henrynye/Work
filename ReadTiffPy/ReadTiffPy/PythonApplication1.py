from  PIL import Image
import numpy
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
folder = "C:\\Users\\henry\\AppData\\Local\\Packages\\Microsoft.MicrosoftEdge_8wekyb3d8bbwe\\TempState\\Downloads\\Food\\Fish_Dual_Energy"
fn0 = ["Fish_20141020_131134_LE","Fish_20141020_131134_HE","Fish_20141020_131134_LE","Fish_20141020_131134_TE"]
fn = folder + "\\" + fn0[0] + ".tif"
im2 = mpimg.imread(fn)
im1 = mpimg.imread(folder + "\\" + fn0[1] + ".tif")
npa1 = numpy.array(im1)
#numpy.resize(npa1,(100,100))
npa1 = numpy.log(npa1)
npa2 = numpy.array(im2)
npa2 = numpy.log(npa2)
#training
box1 = npa1[1200:1520,0:1520]
box2 = npa2[1200:1520,0:1520]
x = numpy.reshape(box2[0:1,800:1500], -1)
y = numpy.reshape(box1[0:1,800:1500], -1)
model = pylab.polyfit(x,y,1)

#plot
v = pylab.polyval(model,x)
pylab.scatter(x,y)
pylab.plot(x,v,"-g")
color = 'red'
pylab.scatter(x,v,c=color)

pylab.text(8.1,8.4,model)
#pylab.show()

#apply model
xnew = numpy.reshape(npa2, -1)
ypred = pylab.polyval(model,xnew)
boxpred = numpy.reshape(ypred, npa2.shape) 
bx = npa1 - boxpred

#plt.imshow(im1)
#plt.show()
#plt.imshow(im2)
#plt.show()
#plt.imshow(box1)
#plt.show()
#plt.imshow(boxpred)
#plt.show()
#plt.imshow(bx)
#plt.show()
box = 1
height = len(bx)
width = len(bx[0])
boxwidth = 2 * box + 1

for k in range(0,4):
    bx = npa1 - boxpred
    for i in range(0,k):
        numpy.rot90(bx)
    plt.imshow(bx)
    plt.show()
    for i in range(box, height - box):
        for j in range(box, width - box):
            medlist = []
            for a in range(0, boxwidth):
                firstD = bx[i - box + a]
                for b in range(0,boxwidth):
                    medlist.append(firstD[j - box + b])
            medianix = int((len(medlist) - 1) / 2)
            medlist.sort()
            bx[i][j] = medlist[medianix]
        print(i , "/" , height - box)
    plt.imshow(bx)
    plt.show()



