from cv2 import imread
from numpy import array, int8, uint8
import numpy
from UZ_utils import *
from druga import * 


##Opening - najprej naredimo erozijo, nato pa še dilation
##Closing - najprej dilation potem se erosion
def prva(n):
    fig = plt.figure(figsize=(5, 5))
    I = imread('images/mask.jpg')


    SE = np.ones((n, n), np.uint8)
    fig.add_subplot(2, 3, 1)
    I_eroded = cv2.erode(I, SE)
    plt.imshow(I_eroded)
    plt.title('Erosion')

    fig.add_subplot(2, 3, 2)
    I_dilated = cv2.dilate(I, SE)
    plt.imshow(I_dilated)
    plt.title('Dilation')

    fig.add_subplot(2, 3, 3)
    Opening = cv2.dilate(I_eroded, SE)

    plt.imshow(Opening)
    plt.title('Opening')

    fig.add_subplot(2, 3, 4)
    Closing = cv2.erode(I_dilated, SE)
    plt.imshow(Closing)
    plt.title('Closing')

    fig.add_subplot(2, 3, 5)
    Opening = cv2.dilate(Opening, SE)
    OandC = cv2.erode(Opening, SE)

    plt.imshow(OandC)
    plt.title('Opening + Closing')

    plt.show()

    return 0

##Izvedena operacija closing 
def sec():
    mask = otsu(imread_gray('images/bird.jpg'))
    mask = mask.astype('uint8')

    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    mask = cv2.dilate(mask, SE)
    mask = cv2.erode(mask, SE)

    plt.imshow(mask, cmap='gray')
    plt.show()
    
    return 0


## Pomnozena binarna maska slike z original sliko
## Crna im vrednost 0, bela pa 1 zato lahko pomnožimo istoležne "prostorcke" in dobimo željen rezultat
def immask(slika, maska):
    X = np.zeros_like(slika)
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    maska = maska.astype('uint8')
    maska = cv2.dilate(maska, SE)
    maska = cv2.erode(maska, SE)

    X[:,:,0] = maska
    X[:,:,1] = maska
    X[:,:,2] = maska
    X = X[:][:][:] * slika[:][:][:]

    plt.imshow(X)
    plt.show()

    return 0

def fourth(slika, maska):
    X = np.zeros_like(slika)
    SE = np.ones((20, 20), np.uint8)
    maska = maska.astype('uint8')
    maska = cv2.erode(maska, SE)
    maska = cv2.dilate(maska, SE)

    X[:,:,0] = maska
    X[:,:,1] = maska
    X[:,:,2] = maska
    X = X[:][:][:] * slika[:][:][:]

    plt.imshow(X)
    plt.show()

    return 0


## Izbrisani elementi na sliki, ki so vecji od 700 pixlov
def last(slika):

    X = otsu(imread_gray('images/coins.jpg'))
    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(1, 2, 1)

    plt.imshow(slika)
    plt.title('image')
    

    SE = np.ones((22, 22), np.uint8)
    X = X.astype('uint8')
    X = cv2.erode(X, SE)

    X = cv2.dilate(X, SE)

    X = np.where(X == 0, 1, 0)

    X = X.astype('uint8')

    nb_components, output, stats, centroids= cv2.connectedComponentsWithStats(X, connectivity=8)
    sizes = stats[1:, -1]

    for i in range(0, nb_components - 1):
        if sizes[i] > 700:
            X[output == i + 1] = 0

    
    fig.add_subplot(1, 2, 2)
    plt.imshow(X, cmap='gray')
    plt.title('result')
    plt.show()
    
    return 0

#a)
#prva(7)

#b)
#sec()

#c)
#immask(imread('images/bird.jpg'), otsu(imread_gray('images/bird.jpg')))

#d)
#fourth(imread('images/eagle.jpg'), otsu(imread_gray('images/eagle.jpg')))

#e)
#last(imread('images/coins.jpg'))