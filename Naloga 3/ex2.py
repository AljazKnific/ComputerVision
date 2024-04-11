from UZ_utils import *
from a3_utils import *
import numpy as np
from ex1 import gradient_magnitude


## Opravljena tocka a)
def findedges(I, sigma, theta):
    mag = gradient_magnitude(I, sigma)
    I_mag = mag[0]

    I_mag = np.where(I_mag >= theta, 1, 0)

    return I_mag

## Opravljena tocka b)
def non_maxima(I, sigma):
    mag = gradient_magnitude(I, sigma)
    angels = mag[1]
    I_mag = mag[0]
    #X =  findedges(I, 1, 0.16)
    #out = X.copy()
    out = mag[0].copy()
    angels = angels + np.pi/8
    angels = np.where(angels < 0, angels + np.pi, angels)
    angels = np.where(angels > np.pi, angels - np.pi, angels)

    visina, sirina = I_mag.shape

    for x in range(1, visina - 1):
        for y in range(1, sirina - 1):
            value = I_mag [x,y]
            kot = angels[x,y]
            if 0 <= kot and kot < np.pi/4:
                if value < I_mag [x, y-1] or value < I_mag [x, y+1]:
                    out[x,y] = 0

            if np.pi/4 <= kot and kot < np.pi/2:
                if value < I_mag [x - 1, y-1] or value < I_mag [x + 1, y+1]:
                    out[x,y] = 0

            if np.pi/2 <= kot and kot < np.pi *3/4:
                if value < I_mag [x-1, y] or value < I_mag [x+1, y]:
                    out[x,y] = 0

            if np.pi * 3/4 <= kot and kot < np.pi:
                if value < I_mag [x-1, y+1] or value < I_mag[x +1 , y-1]:
                    out[x,y] = 0


    return out

## Opravljena tocka c)
def hysteresis(high, low, I, sigma):
    Rez = np.zeros((I.shape[0], I.shape[1]))
    X = non_maxima(I, sigma)
    X_low = np.where(X > low, 1, 0)
    X_high = np.where(X > high, 1, 0)


    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(X_low.astype('uint8'), connectivity=8)
    #print(nb_components)
    #print(output)
    #print(stats)
    #print(centroids)

    for i in range(0, nb_components - 1):
        max = np.max(X_high[output == i])
        if max == 1:
            Rez[output == i] = 1

    return Rez
## Prikaz vseh slik podane naloge
def prikaz_muzejev():
    sigma = 1
    I = imread_gray('images/museum.jpg')
    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2,2, 1, title='Original')
    plt.imshow(I, cmap='gray')

    fig.add_subplot(2, 2, 2, title='Theta = 0.16')
    plt.imshow(findedges(I, sigma, 0.16), cmap='gray')

    fig.add_subplot(2, 2, 3, title='Non-maxima (Theta = 0.16)')
    X = non_maxima(I, sigma)
    X = np.where(X >= 0.16, 1, 0)
    plt.imshow(X, cmap='gray')

    fig.add_subplot(2, 2, 4, title='Hysteresis (high = 0.16, low = 0.04)')
    M = hysteresis(0.16, 0.04, I, sigma)
    plt.imshow(M, cmap='gray')

    plt.show()

    return 0

prikaz_muzejev()