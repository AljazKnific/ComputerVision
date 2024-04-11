from numpy import array, int8, uint8
import numpy
from UZ_utils import *


## Z vgnezdenim for loopom grem cez celotno matriko
## sestevam RGB kanale in jih delim s 3 in rezultat
## zapišem na enako mesto v drugo matriko
def sivo(slika):
    I = imread(slika)
    visina, sirina, channels = I.shape
    X = np.zeros([visina,sirina], float)
    for i in range(0, visina):
        for j in range(0, sirina):
            red, green, blue = I[i][j]
            X[i][j] = ((red + green + blue) / 3) 

    plt.imshow(X, cmap="gray")
    plt.show()

    return 0

## Iz original slike si izberem izsek
## katerega spremenim v grayscale
def izsek(xP, xD , yP, yD , slika):
    fig = plt.figure(figsize=(5, 5))
    I = imread(slika)
    cutout = I[xP:xD, yP:yD, 1]

    fig.add_subplot(1, 2, 1)
    plt.imshow(I)
    plt.title('Original slika')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(cutout, cmap='gray')
    plt.title('Izsek slike ' + str(xP) + ':' + str(xD) + ' , ' +  str(yP) + ':' + str(yD))

    plt.show()

    return 0


## Določen del iz slike zamenjam z njegovim komplementom
def menjavaDela(xP, xD , yP, yD , slika):
    I = imread(slika)
    for i in range(xP, xD):
        for j in range(yP, yD):
            I[i][j][0] = 1 - I[i][j][0]
            I[i][j][1] = 1 - I[i][j][1]
            I[i][j][2] = 1 - I[i][j][2]

    plt.imshow(I)
    plt.show()

    return 0



def zadnja(slika):
    fig = plt.figure(figsize=(5, 5))
    I = imread_gray(slika)
    X = imread_gray(slika)

    
    fig.add_subplot(1, 2, 1)

    plt.imshow(X, cmap='gray')
    plt.title('Prva slika')

    visina, sirina = I.shape
    
    L = np.zeros([visina,sirina], uint8)

    for i in range(0,visina):
        for j in range(0, sirina):
            L[i][j] = int(I[i][j] * 63)


    fig.add_subplot(1, 2, 2)
    plt.imshow(L, cmap='gray', vmax=255)
    plt.title('Druga slika')
    
    plt.show()

    return 0

#a)
#I = imread('images/umbrellas.jpg')
#imshow(I)

#b) 
#sivo('images/umbrellas.jpg')

#c)
#izsek(130, 240, 230 , 440, 'images/umbrellas.jpg')

#d)
#menjavaDela(130, 240, 230 , 440, 'images/umbrellas.jpg')

#e)
zadnja('images/umbrellas.jpg')

