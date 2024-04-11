from UZ_utils import *
from a3_utils import *
import numpy as np

## Funckija za racunanje gaussovega kernela
def gaus_ker(sigma):
    vel = int((np.ceil(sigma * 3) * 2) + 1)
    kernal = np.zeros((1,vel))
    
    gor = int(np.ceil(vel/2))
    dol = int(-np.floor(vel/2))


    for x in range(dol, gor):
        rez = (1 / (np.sqrt(2 * np.pi) * sigma)) 
        e = np.exp(-((x**2)/(2 * sigma**2)))
        funk = rez * e
        kernal[0,x - dol] = funk

    kernal = kernal / np.sum(kernal)
    return kernal

## Funckija za racunanje gauss derivate kernela
def gaussdx(sigma):
    vel = int((np.ceil(sigma * 3) * 2) + 1)
    kernal = np.zeros((1,vel))
    stevec = 0
    
    gor = int(np.ceil(vel/2))
    dol = int(-np.floor(vel/2))

    for x in range(dol, gor):
        rez = -(1 / (np.sqrt(2 * np.pi) * sigma**3))
        e = np.exp(-(x**2)/(2 * sigma**2))
        funk = rez * e * x
        kernal[0,x - dol] = funk

        stevec += np.abs(funk)

    kernal = kernal / stevec

    return kernal

## Funkcija za racunanje odvodov -> Dolocis nacin glede na tip odvoda
def odvod(I, sigma, nacin):
    G = gaus_ker(sigma)
    GD = gaussdx(sigma)
    if nacin == 'X':
        I = cv2.filter2D(cv2.filter2D(I, -1, G.T), -1, np.flip(GD))
    if nacin == 'Y':
        I = cv2.filter2D(cv2.filter2D(I, -1, G), -1, np.flip(GD.T))
    return I

## Funckija za racunanje magnitud in kotov
def gradient_magnitude(I, sigma):
    I_x = odvod(I, sigma, 'X')
    I_y = odvod(I, sigma, 'Y')
    magnitude = np.sqrt(np.multiply(I_x, I_x) + np.multiply(I_y, I_y))
    angles = np.arctan2(I_y, I_x)
    return magnitude, angles


## Izpis slicic c) naloge pri prvi vaji
def prva():
    impulse = np.zeros((50,50))
    impulse[25,25] = 255

    gaus_1d = gaus_ker(5)
    gaus_der = gaussdx(5)

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2, 3, 1)
    plt.imshow(impulse, cmap='gray')
    plt.title('Impulse')


    fig.add_subplot(2, 3, 2)
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, gaus_1d), -1, np.flip(gaus_der.T)), cmap='gray')
    plt.title('G, Dt')


    fig.add_subplot(2, 3, 3)
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, np.flip(gaus_der)), -1, gaus_1d.T), cmap='gray')
    plt.title('D, Gt')


    fig.add_subplot(2, 3, 4)
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, gaus_1d), -1, gaus_1d.T), cmap='gray')
    plt.title('G, Gt')

    fig.add_subplot(2, 3, 5)
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, gaus_1d.T), -1, np.flip(gaus_der)), cmap='gray')
    plt.title('Gt, D')

    fig.add_subplot(2, 3, 6)
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, np.flip(gaus_der.T)), -1, gaus_1d), cmap='gray')
    plt.title('Dt, G')

    plt.show()

    
    return 0 

## Izris slicic d) naloge pri prvi vaji
def druga(I):
    sigma = 1.1
    I_x = odvod(I, sigma,'X')
    I_y = odvod(I, sigma, 'Y')
    I_xx = odvod(I_x, sigma, 'X')
    I_yy = odvod(I_y, sigma,  'Y')
    I_xy = odvod(I_x, sigma,  'Y')
    mag = gradient_magnitude(I, sigma)

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2, 4, 1, title='Original')
    plt.imshow(I, cmap='gray')

    fig.add_subplot(2, 4, 2, title='I_x')
    plt.imshow(I_x, cmap='gray')

    fig.add_subplot(2, 4, 3, title='I_y')
    plt.imshow(I_y, cmap='gray')

    fig.add_subplot(2, 4, 4, title='I_mag')
    plt.imshow(mag[0], cmap='gray')

    fig.add_subplot(2, 4, 5, title='I_xx')
    plt.imshow(I_xx, cmap='gray')

    fig.add_subplot(2, 4, 6, title='I_xy')
    plt.imshow(I_xy, cmap='gray')

    fig.add_subplot(2, 4, 7, title='I_yy')
    plt.imshow(I_yy, cmap='gray')

    fig.add_subplot(2, 4, 8, title='I_dir')
    plt.imshow(mag[1], cmap='gray')

    plt.show()

    return 0

#prva()
#druga(imread_gray('images/museum.jpg'))