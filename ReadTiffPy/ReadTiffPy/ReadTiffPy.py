

import numpy
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.stats
import statistics
import random
import math
from scipy.ndimage.filters import gaussian_filter
from PIL import *
import PIL.Image as imx
 

##training

##plot
#v = pylab.polyval(model,x)
#pylab.scatter(x,y)
#pylab.plot(x,v,"-g")
#color = 'red'
#pylab.scatter(x,v,c=color)

#pylab.text(8.1,8.4,model)
##pylab.show()

##apply model

#plt.imshow(bx)
#plt.show()

#height = len(bx)
#width = len(bx[0])
#box = 1




def openimage(img):
    folder = "C:\\Users\\henry\\OneDrive\\Documents\\Data"
    fn0 = ["Fish_20141020_131134_LE","Fish_20141020_131134_HE","Fish_20141020_131134_TE","XcTomoFliteImage_20141002_144854_LE","XcTomoFliteImage_20141002_144854_HE","bones fish, box size = 3X3, Filter Type complex, Noise Standard Deviation 50"]
    fn = folder + "\\" + fn0[img] + ".tif"
    im1 = mpimg.imread(fn)
    #mpimg.imsave("a",im1)
    npa1 = numpy.array(im1)
    finalimg = npa1[1200:1520,0:1800]
    print("opened")
    return npa1

def addnoise(img, noise):
    img2 = numpy.copy(img)
    for i in range(0,len(img2)):
        for j in range(0,len(img2[i])):
            number = numpy.random.normal(loc = 0,scale = noise)
            img2[i][j] += number
    return img2
def medfilter(img1):

    img3 = numpy.copy(img1)
    box = size
    boxwidth = 2 * box + 1
    height = len(img3)
    width = len(img3[0])
    for i in range(box, height - box):
        for j in range(box,width - box):
            medlist = []
            for a in range(0, boxwidth):
                firstD = img1[i - box + a]
                for b in range(0,boxwidth):
                    medlist.append(firstD[j - box + b]) 
            medianix = int((len(medlist) - 1) / 2)
            medlist.sort() 
            img3[i][j] = medlist[medianix]
        print(i , "/" , height - box)
    return img3
def imgdifference(img1, img2):
    cleanfilter = medfilter(img1)
    noiseyfilter = medfilter(img2)
    difference = noiseyfilter - cleanfilter
    plt.imshow(cleanfilter)
    plt.show()
    plt.imshow(noiseyfilter)
    plt.show()
    return difference
def meansquarederror(img1, img2):
    sqrsum = 0.0
    for i in range(0,len(img1)):
            k = img2[i] - img1[i]
            sqrsum += (k**2)
    mnsqr = math.sqrt(sqrsum /(len(img1)))
    print("Mean Squared Error: ",mnsqr)
    return mnsqr
def training(base, target):
    basecopy = numpy.copy(base)
    targetcopy = numpy.copy(target)
    box1 = basecopy
    box2 = targetcopy
    x = numpy.reshape(box1[0:1,800:1500], -1)
    y = numpy.reshape(box2[0:1,800:1500], -1)
    model = pylab.polyfit(x,y,1)
    return model
def applymodel(model, baseimage, targetimage):
    xnew = numpy.reshape(baseimage, -1)
    ypred = pylab.polyval(model,xnew)
    boxpred = numpy.reshape(ypred, baseimage.shape) 
    bx = targetimage - boxpred
    return bx
def display(img):
    plt.imshow(img)
    plt.show()
def compmedfilter(img):
    img2 = numpy.copy(img)
    box = size
    boxwidth = 2 * box + 1
    height = len(img2)
    width = len(img2[0])
    bx2 = []
    for k in range (0,4):
        bx = numpy.copy(img2)
        for m  in range (0,k):
            bx= numpy.rot90(bx)
        heightrot = len(bx)
        widthrot = len(bx[0])
        for i in range(box,heightrot - box):
            for j in range(box,widthrot - box):
                medlist = []
                for a in range(0, boxwidth):
                    firstd = bx[i - box + a]
                    for b in range(0,boxwidth):
                        medlist.append(firstd[j - box + b])
                medianix = int((len(medlist) - 1) / 2)
                medlist.sort()
                bx[i][j] = medlist[medianix]
            print(i , "/" , heightrot - box)
        for n in range(0,4-k):
            bx = numpy.rot90(bx)
        bx2.append(bx)
    bx3 = numpy.copy(bx)
    for i in range(box, height - box):
        for j in range(box,width - box):
            medlist = []
            for k in bx2:
                medlist.append(k[i][j])
                medlist.sort()
                medianix = int((len(medlist) - 1) / 2)
                bx3[i][j] = medlist[medianix]
    return bx3
def mask(img):
    img2 = numpy.copy(img)
    standarddev = numpy.std(img2)
    coords = []
    for i in range(0,len(img2)):
        for j in range(0,len(img2[i])):
            if img2[i][j] < (0-(standarddev*6)):
                img2[i][j] = 1
                coords.append([i,j])
                img2[i][j]+=5
            else:
                img2[i][j] = 0
    return(coords)
def simpplot(badimg, goodimg):
    plt.close()
    x = numpy.reshape(goodimg, -1)
    y = numpy.reshape(badimg, -1)
    pylab.scatter(x,y)
    snr = signaltonoise(x,y)
    mnsqrerror = meansquarederror(x,y)
    plt.title("MSE: "+ str(round(mnsqrerror,2))+"\n"+"SNR: "+str(round(snr,2)))
    plt.xlabel("Gold Standard Pixel Value")
    plt.ylabel("Noisy Pixel Value")
    plt.savefig(kname)
    #pylab.show()
def signaltonoise(indep, depen):
    n = len(indep)
    
    meanx = numpy.mean(indep)
    meany = numpy.mean(depen)
    stdevx = numpy.std(indep)
    stdevy = numpy.std(depen)
    cov = 0.0
    for i in range(0,len(indep)):
        cov+=((indep[i]-meanx)*(depen[i]-meany))
    cov /= (n)
    corco = cov/(stdevx*stdevy)
    print("Correlation Coefficient: ",corco)
    signoise=corco/(1-corco)
    print("Signal To Noise: ",signoise)
    return signoise
def meanfilter(img1):

    img3 = numpy.copy(img1)
    box = size
    boxwidth = 2 * box + 1
    height = len(img3)
    width = len(img3[0])
    for i in range(box, height - box):
        for j in range(box,width - box):
            meanlist = []
            for a in range(0, boxwidth):
                firstD = img1[i - box + a]
                for b in range(0,boxwidth):
                    meanlist.append(firstD[j - box + b])
            msum = sum(meanlist)
            mean = msum/len(meanlist)
            img3[i][j] = mean
        print(i , "/" , height - box)
    return img3
def saveimg(img,imgname):    
    z=img *1.0
    x=imx.fromarray(z)
    fn = imgname
    x.save(fn)
def opendifimg(img):
    xf = open(img,'rb')
    xx = imx.open(xf)
    xxx = numpy.array(xx)
    return xxx

#baseimg = openimage(1)
#targetimg = openimage(0)

#noiseybase = addnoise(baseimg,100)
#noiseytarget = addnoise(targetimg,100)

##filtbase = numpy.log(filtbase) 
##filttarget = numpy.log(filttarget)
#noiseybase = numpy.log(noiseybase)
#noiseytarget = numpy.log(noiseytarget)
#baseimg = numpy.log(baseimg)
#targetimg = numpy.log(targetimg)


##filtbase = medfilter(baseimg)
##filttarget = medfilter(targetimg)
##filtnoiseybase = medfilter(noiseybase)
##filtnoiseytarget = medfilter(noiseytarget)

#baseimg = baseimg[900:1300, 0:1800]
#targetimg = targetimg[900:1300, 0:1800]
#noiseybase = noiseybase[900:1300, 0:1800]
#noiseytarget = noiseytarget[900:1300, 0:1800]

##display(filtbase)
##display(filttarget)
##display(filtnoiseybase)
##display(filtnoiseytarget)

#xmodel = training(baseimg, targetimg)
#bones = applymodel(xmodel, baseimg, targetimg)

#badmodel = training(noiseybase,noiseytarget)
#noiseybones = applymodel(badmodel, noiseybase, noiseytarget)

###filtgold = compmedfilter(otherimg)
###filtnoise = compmedfilter(noisey)


#display(bones)
#display(noiseybones)
##filta = medfilter(noiseybones)
##filtb = medfilter(bones)
##display(filtb)
##display(filta)


#simpplot(noiseybones, bones)

################################################
def script(img, bxsize, filttype, noisestdev):
    global size
    size = bxsize
    baseimg = "x"
    targetimg = "y"
    if img == "fish":
        baseimg = openimage(1)
        targetimg = openimage(0)

    noisybase = addnoise(baseimg, noisestdev)
    noisytarget = addnoise(targetimg, noisestdev)
    
    #display(baseimg)
    #display(targetimg)
    #display(noisybase)
    #display(noisytarget)

    noisybase = numpy.log(noisybase)
    noisytarget = numpy.log(noisytarget)

    noisybase = noisybase[900:1300, 0:1800]
    noisytarget = noisytarget[900:1300, 0:1800]
    if filttype == "complex":
        filtnoisybase = compmedfilter(noisybase)
        filtnoisytarget = compmedfilter(noisytarget)
    elif filttype == "simple":
        filtnoisybase = medfilter(noisybase)
        filtnoisytarget = medfilter(noisytarget)
    elif filttype == "mean":
        filtnoisybase = meanfilter(noisybase)
        filtnoisytarget = meanfilter(noisytarget)

    #display(filtbase)
    #display(filttarget)
    #display(filtnoisybase)
    #display(filtnoisytarget)


    badmodel = training(filtnoisybase,filtnoisytarget)
    noisybones = applymodel(badmodel, filtnoisybase, filtnoisytarget)

    
    #display(bones)
    #display(noisybones)


    #noisybones = medfilter(noisybones)
     #bones = medfilter(bones)

    #display(noisybones)
    #display(bones)
    global kname 
    fname = "C:/Users/henry/OneDrive/Documents/Summer Work Images/Code Written Images/"
    details = (str(img) + ", box size = " +str((bxsize*2)+1)+"X"+str((bxsize*2)+1)+", Filter Type " + str(filttype) +", Noise Standard Deviation "+ str(noisestdev))
    iname = (fname + "noisy bones " + details + ".tif")    
    jname = (fname + "bones fish, box size = 3X3, Filter Type complex.tif")
    kname = (fname + "plot " + details + ".tif")
   
    bones = opendifimg(jname)
    simpplot(noisybones[:,600:800], bones[:,600:800])
    mpimg.imsave(iname, noisybones)

    

noises = [50,100,150,200,250,300]
images = ["fish","chicken"]
filters = ["complex", "simple", "mean"]
for noise in noises:
    noisestdev = noise
    for filter in filters:
        filtertype = filter
        for i in range(1,3):
            image = "fish"
            boxsize = i #int((int(input("Size of filter box (NxN, Enter N): "))-1)/2)
            script(image, boxsize, filtertype, noisestdev)



