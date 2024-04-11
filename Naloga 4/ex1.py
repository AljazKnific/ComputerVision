from UZ_utils import *
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



def hessian_points(I,sigma, threshold):
    I_x = odvod(I, sigma, 'X')
    I_y = odvod(I, sigma, 'Y')
    I_xx = odvod(I_x, sigma, 'X')
    I_yy = odvod(I_y, sigma, 'Y')
    I_xy = odvod(I_x, sigma, 'Y')

    rez = I_xx * I_yy
    rez2 = I_xy * I_xy

    rez = rez - rez2
    X = non_maximum_suppression(rez)
    X = np.argwhere(X > threshold)

    return rez, X

def non_maximum_suppression(I):
    res = I.copy()

    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            sez = []

            if i == 0:
                if j == 0:
                    sez.append(I[i, j + 1])
                    sez.append(I[i + 1, j])
                    sez.append(I[i + 1, j + 1])
            #pogledamo ce je prva vrstica
                elif 0 < j and j < I.shape[1] - 1:
                    sez.append(I[i, j - 1])
                    sez.append(I[i, j + 1])
                    sez.append(I[i + 1, j - 1])
                    sez.append(I[i + 1, j])
                    sez.append(I[i + 1, j + 1])

                elif j == I.shape[1] - 1:
                    sez.append(I[i, j - 1])
                    sez.append(I[i + 1, j - 1])
                    sez.append(I[i + 1, j])

            elif 0 < i and i < I.shape[0] - 1:
                if j == 0:
                    sez.append(I[i - 1, j])
                    sez.append(I[i - 1, j + 1])
                    sez.append(I[i, j + 1])
                    sez.append(I[i + 1, j])
                    sez.append(I[i + 1, j + 1])

                elif 0 < j and j < I.shape[1] - 1:
                    sez.append(I[i - 1, j - 1])
                    sez.append(I[i - 1, j])
                    sez.append(I[i - 1, j + 1])

                    #middle vrstica
                    sez.append(I[i, j - 1])
                    sez.append(I[i, j + 1])

                    #spodnja vrstica
                    sez.append(I[i + 1, j - 1])
                    sez.append(I[i + 1, j])
                    sez.append(I[i + 1, j + 1])

                elif j == I.shape[1] - 1:
                    sez.append(I[i - 1, j - 1])
                    sez.append(I[i - 1, j])
                    sez.append(I[i, j - 1])
                    sez.append(I[i + 1, j - 1])
                    sez.append(I[i + 1, j])

            elif i == I.shape[0] - 1:
                if j == 0:
                    sez.append(I[i - 1, j])
                    sez.append(I[i - 1, j + 1])
                    sez.append(I[i, j + 1])

                elif 0 < j and j < I.shape[1] - 1:
                    sez.append(I[i - 1, j - 1])
                    sez.append(I[i - 1, j])
                    sez.append(I[i - 1, j + 1])

                    #middle vrstica
                    sez.append(I[i, j - 1])
                    sez.append(I[i, j + 1])

                elif j == I.shape[1] - 1:
                    sez.append(I[i - 1, j - 1])
                    sez.append(I[i - 1, j])
                    sez.append(I[i, j - 1])

            if I[i,j] <= np.max(sez):
                res[i,j] = 0

    return res

def harris_point_det(I, sigma, threshold, alpha):
    I_x = odvod(I, sigma, 'X')
    I_y = odvod(I, sigma, 'Y')

    temp1 = I_x * I_x
    temp2 = I_y * I_y
    temp3 = I_x * I_y

    kernal = gaus_ker(sigma * 1.6)

    C_11 = cv2.filter2D(temp1, -1, kernal.T)
    C_22 = cv2.filter2D(temp2, -1, kernal.T)
    C_11 = cv2.filter2D(C_11, -1, kernal)
    C_22 = cv2.filter2D(C_22, -1, kernal)

    odstejes = cv2.filter2D(temp3, -1, kernal.T)
    odstejes = cv2.filter2D(odstejes, -1, kernal)

    det = C_11 * C_22 - odstejes**2
    trace = C_11 + C_22
    trace = trace * trace

    rez = det - alpha*trace

    X = non_maximum_suppression(rez)
    X = np.argwhere(X > threshold)

    return rez, X

def prva():
    I = imread_gray('data/graf/graf_a.jpg')
    I1 = hessian_points(I, 3, 0.006)
    I2 = hessian_points(I, 6, 0.004)
    I3 = hessian_points(I, 9, 0.004)


    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(2, 3, 1)
    plt.imshow(I1[0])
    plt.title('sigma = 3')

    fig.add_subplot(2, 3, 2)
    plt.imshow(I2[0])
    plt.title('sigma = 6')

    fig.add_subplot(2, 3, 3)
    plt.imshow(I3[0])
    plt.title('sigma = 9')

    fig.add_subplot(2, 3, 4)
    plt.imshow(I, cmap='gray')


    plt.scatter(I1[1][:,1], I1[1][:,0], marker='x', color='red')

    fig.add_subplot(2, 3, 5)
    plt.imshow(I, cmap='gray')


    plt.scatter(I2[1][:,1], I2[1][:,0], marker='x', color='red')

    fig.add_subplot(2, 3, 6)
    plt.imshow(I, cmap='gray')


    plt.scatter(I3[1][:,1], I3[1][:,0], marker='x', color='red')

    plt.show()
    return 0

def druga():

    I = imread_gray('data/graf/graf_a.jpg')
    I1 = harris_point_det(I, 3, 0.000001, 0.06)   
    I2 = harris_point_det(I, 6, 0.000001, 0.06)
    I3 = harris_point_det(I, 9, 0.000001, 0.06)


    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(2, 3, 1)
    plt.imshow(I1[0])
    plt.title('sigma = 3')

    fig.add_subplot(2, 3, 2)
    plt.imshow(I2[0])
    plt.title('sigma = 6')

    fig.add_subplot(2, 3, 3)
    plt.imshow(I3[0])
    plt.title('sigma = 9')

    fig.add_subplot(2, 3, 4)
    plt.imshow(I, cmap='gray')


    plt.scatter(I1[1][:,1], I1[1][:,0], marker='x', color='red')

    fig.add_subplot(2, 3, 5)
    plt.imshow(I, cmap='gray')


    plt.scatter(I2[1][:,1], I2[1][:,0], marker='x', color='red')

    fig.add_subplot(2, 3, 6)
    plt.imshow(I, cmap='gray')


    plt.scatter(I3[1][:,1], I3[1][:,0], marker='x', color='red')

    plt.show()

    return 0

#prva()
#druga()

## Questions:
#What kind of structures in the image are detected by the algorithm?
# How does the parameter σ affect the result?
# It detects blobs.
# The bigger the sigma less points it detects.
# Hessian algorithm -> blobs, Harris algorithm -> corners
