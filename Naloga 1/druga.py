from ctypes import sizeof
from unittest import result
from cv2 import IMREAD_GRAYSCALE
from UZ_utils import *

## Imamo 2 načina kreacije binarne maske. 
# 1. način predstavlja spremembo pixlov v vrednosti 0 ali 1, glede na to
# ali je pixel v binarni sliki pod ali nad thresholdom. Z 2ma for loopoma gremo čez
# matriko in določamo pixle.
# 2. Način predstavlja isti koncept, vendar smo za izvedbo porabili samo 1 ukaz

def aDruga(slika, x, nacin):
    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(1, 2, 1)
    I = imread_gray(slika)
    plt.imshow(I, cmap='gray')
    plt.title('Image')

# 1.Način
    if nacin == 0:
        visina, sirina = I.shape

        for i in range(0, visina):
            for j in range(0, sirina):
                if I[i][j] > x:
                    I[i][j] = 1
                else:
                    I[i][j] = 0

        fig.add_subplot(1, 2, 2)

        plt.imshow(I, cmap='gray')
        plt.title('Maska')
    
# 2.Način
    else:
        L = imread_gray(slika)

        L = np.where(L > x, 1 , 0)

        fig.add_subplot(1, 2, 2)

        plt.imshow(L, cmap='gray')
        plt.title('Maska')

    plt.show()
    return 0


## Implementacija b točke
# Imamo 2 nacina
# Nacin 0 vzame za max = 255 in min = 0
# Nacin 1 pa vzame za max vrednosti ki je med podanimi najvecja, min pa med podanimi najmanjso
def myhist(slika, bins, nacin):

    H = np.zeros(bins)
    min=0
    max=255

    if nacin == 1:
        min = np.min(slika) * 255
        max = np.max(slika) * 255

    #print(min, max)
    velikost = (max - min) / bins


    slika = slika.reshape(-1)
    vis= slika.shape

    for j in range(0,vis[0]):
        stevec = 0
        x = velikost + min

        for i in range(0,bins - 1):
            if (slika[j] * 255) > x:
                stevec = stevec + 1
                x = x + velikost
            else:
                break

        H[stevec] = H[stevec] + 1

    H = H / np.sum(H)

    return H


## Prezentacija histogramov 3 različno osvetljenih slik
def tretja(bins):
    tema = imread_gray('slike/brez.jpg')
    lucka = imread_gray('slike/lucka.jpg')
    luc = imread_gray('slike/luc.jpg')

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2, 3, 1)

    X = myhist(tema, bins, 1)

    plt.bar(range(bins), X)
    plt.title('Temna slika')

    
    fig.add_subplot(2, 3, 2)

    Y = myhist(lucka, bins, 1)

    plt.bar(range(bins), Y)
    plt.title('Srednje svetla slika')

    fig.add_subplot(2, 3, 3)

    Z = myhist(luc, bins, 1)

    plt.bar(range(bins), Z)
    plt.title('Svetla slika')

    fig.add_subplot(2, 3, 4)
    plt.imshow(tema, cmap='gray')
    fig.add_subplot(2, 3, 5)
    plt.imshow(lucka, cmap='gray')
    fig.add_subplot(2, 3, 6)
    plt.imshow(luc, cmap='gray')

    plt.show()

## zapisana otsu metoda po formulah
def otsu(pic):

    fig = plt.figure(figsize=(5, 5))
    max_varianca = 0
    bins = 255
    Histo = myhist(pic, bins, 0)
    dol = Histo.shape[0]
    index = 0

    tabela = []
    
    for i in range(0, dol):
        gamaZero = 0
        gamaOne = 0
        miZero = 0
        miOne = 0


        for j in range(0, dol):
            if j < i:
                gamaZero = gamaZero + Histo[j]
                miZero = miZero + (Histo[j] * j)
            else:
                gamaOne = gamaOne + Histo[j]
                miOne = miOne + (Histo[j] * j)

        if gamaZero != 0:
            miZero = miZero / gamaZero
        else: 
            miZero = 0

        if gamaOne != 0:
            miOne = miOne / gamaOne
        else: 
            miOne = 0


        result = pow((miOne - miZero), 2)
        tabela.append(gamaOne * gamaZero * result)

## gledam kdaj je varianca najvecja in si shranim pravi index
        if (gamaOne * gamaZero * result) > max_varianca:
            max_varianca = gamaOne * gamaZero * result
            index = i
    
## izracunam threshold glede na bin in index
    vel = (1 / bins) * index

    L = np.where(pic > vel, 1 , 0)
    """
    fig.add_subplot(1, 2, 1)

    plt.plot(tabela)
    plt.title('Bins = ' + str(bins) + ' Threshold = ' +  str(index))


    fig.add_subplot(1, 2, 2)
    plt.imshow(L, cmap='gray')
    plt.title('Uporabljena otsu metoda na sliki')
    plt.show()
    """
    return L

#a)
#aDruga('images/bird.jpg', 0.29, 0)

#b)
slikca = imread_gray('images/bird.jpg')

"""
H = myhist(slikca, 100, 0)
plt.bar(range(100), H)
plt.show()
"""

#c)
#myhist(slikca, 100, 0)
#myhist(slikca, 100 , 1)

#d)
tretja(100)
piksa = imread_gray('images/eagle.jpg')
#e)
#otsu(piksa)
