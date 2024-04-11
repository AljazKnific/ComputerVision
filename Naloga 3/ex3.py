from UZ_utils import *
from a3_utils import *
import numpy as np
from ex2 import findedges, non_maxima
from ex1 import gradient_magnitude

def accumulator(bins, x, y):
    Acc = np.zeros((bins, bins))

    ThetaArray = np.linspace(-np.pi/2, np.pi, bins)

    for i in range(0, bins - 1):
        Rez = (x * np.cos(ThetaArray[i])) + (y * np.sin(ThetaArray[i]))

        Acc[int(bins/2 + Rez), i] += 1

    return Acc

def accM(x, y, A, t_bins, r_bins, D):

    ThetaArray = np.linspace(-np.pi/2, np.pi/2, t_bins)

    for i in range(0, t_bins):
        Rez = (y * np.cos(ThetaArray[i])) + (x * np.sin(ThetaArray[i]))

        j = int(((Rez + D) / (2 * D)) * r_bins)

        if 0 <= j and j < r_bins:
            A[j,i] += 1

    return A



def hough_find_lines(I, theta_bins, rho_bins, threshold):
    I = non_maxima(I, 1)
    I = np.where(I > threshold, 1, 0)
    diagonala = np.sqrt(I.shape[0]**2 + I.shape[1]**2)
    A = np.zeros((rho_bins, theta_bins))

    I = np.where(I > 0)
    Prva = I[0]
    Druga = I[1]

    for i in range(0, len(Prva)):
        x = Prva[i]
        y = Druga[i]


        A = accM(x,y,A,theta_bins,rho_bins, diagonala)
    
    return A

def nonmaxima_supression_box(I):
    res = I.copy()

    for i in range(1, I.shape[0] - 1):
        for j in range(1, I.shape[1] - 1):
            sez = []

            sez.append(I[i - 1, j - 1])
            sez.append(I[i - 1, j])
            sez.append(I[i - 1, j + 1])

            sez.append(I[i, j - 1])
            sez.append(I[i, j + 1])

            sez.append(I[i + 1, j - 1])
            sez.append(I[i + 1, j])
            sez.append(I[i + 1, j + 1])

            if I[i,j] < np.max(sez):
                res[i,j] = 0
            
    return res


def threshold_slika(I, theta_bins, rho_bins, thres, origi, thresehold):
    G = origi.copy()
    diagonala = np.sqrt(origi.shape[0]**2 + origi.shape[1]**2)
    I = hough_find_lines(I, theta_bins, rho_bins, thresehold)
    I = nonmaxima_supression_box(I)

    thetaA = np.linspace(- np.pi / 2, np.pi / 2, theta_bins)

    RezSet = np.where(I > thres)

    for i in range(0, len(RezSet[0])):
        x = RezSet[0][i]
        y = RezSet[1][i]
        rho = ((2 * diagonala * x) / rho_bins) - diagonala
        draw_line(rho, thetaA[y], G.shape[0], G.shape[1])
    
    plt.imshow(G, cmap='gray')

    return 0

def prva():
    fig = plt.figure(figsize=(5, 5))

    I = accumulator(300, 10, 10)

    fig.add_subplot(2,2, 1, title='x = 10, y = 10')
    plt.imshow(I)

    I = accumulator(300, 30, 60)

    fig.add_subplot(2,2, 2, title='x = 30, y = 60')
    plt.imshow(I)

    I = accumulator(300, 50, 20)

    fig.add_subplot(2,2, 3, title='x = 50, y = 20')
    plt.imshow(I)

    I = accumulator(300, 80, 90)

    fig.add_subplot(2,2, 4, title='x = 80, y = 90')
    plt.imshow(I)

    plt.show()


    return 0

def druga():
    fig = plt.figure(figsize=(5, 5))
    S = np.zeros((100, 100))
    S[10, 10] = 1
    S[10, 20] = 1
    A = hough_find_lines(S, 200, 200, 0.1)
    
    fig.add_subplot(1,3, 1, title='Synthetic')
    plt.imshow(A)

    
    X = imread_gray('images/oneline.png')
    #B = hough_find_lines(findedges(X, 1, 0.25), 200, 200, 5)
    B = hough_find_lines(X, 200, 200, 0.16)

    fig.add_subplot(1,3, 2, title='oneline.png')
    plt.imshow(B)

    
    X1 = imread_gray('images/rectangle.png')
    #L = hough_find_lines(findedges(X1, 1, 0.25), 200, 200, 5)
    L = hough_find_lines(X1, 200, 200, 0.16)

    fig.add_subplot(1,3, 3, title='rectangle.png')
    plt.imshow(L)


    plt.show()
    return 0

def tretja():
    fig = plt.figure(figsize=(5, 5))
    S = np.zeros((100, 100))
    S[10, 10] = 1
    S[10, 20] = 1
    A = hough_find_lines(S, 200, 200, 0.13)
    
    fig.add_subplot(1,3, 1, title='Synthetic')
    plt.imshow(nonmaxima_supression_box(A))

    X = imread_gray('images/oneline.png')
    #B = hough_find_lines(findedges(X, 1, 0.25), 200, 200, 5)
    B = hough_find_lines(X, 200, 200, 0.16)
    fig.add_subplot(1,3, 2, title='oneline.png')
    prikaz = nonmaxima_supression_box(B)
    plt.imshow(prikaz)
    
    
    X1 = imread_gray('images/rectangle.png')
    #L = hough_find_lines(findedges(X1, 1, 0.25), 200, 200, 5)
    L = hough_find_lines(X1, 200, 200, 0.16)

    fig.add_subplot(1,3, 3, title='rectangle.png')
    plt.imshow(nonmaxima_supression_box(L))
    

    plt.show()
    return 0


def cetrta():
    fig = plt.figure(figsize=(5, 5))
    S = np.zeros((100, 100))
    S[10, 10] = 1
    S[10, 20] = 1

    
    fig.add_subplot(1,3, 1, title='Synthetic')
    threshold_slika(S, 300, 300, 3, S, 0.13)
    
    I2 = imread_gray('images/oneline.png')
    fig.add_subplot(1,3, 2, title='oneline.png')
    #threshold_slika(findedges(I2, 1, 0.25), 300, 300, 500, I2, 0.16)
    threshold_slika(I2, 300, 300, 200, I2, 0.16)
    
    I3 = imread_gray('images/rectangle.png')
    fig.add_subplot(1,3, 3, title='rectangle.png')
    #threshold_slika(findedges(I3, 1, 0.25), 300, 300, 210, I3, 0.16)
    threshold_slika(I3, 300, 300, 200, I3, 0.16)
    
    plt.show()

    return 0

def peta():
    fig = plt.figure(figsize=(5, 5))

    I1G = imread('images/bricks.jpg')
    I1 = imread_gray('images/bricks.jpg')

    fig.add_subplot(2,2, 1, title='bricks.jpg')

    #FB = findedges(I1, 0.75, 0.20)
    #B = hough_find_lines(FB, 300, 300, 0)
    B = hough_find_lines(I1, 300, 300, 0.16)
    plt.imshow(B)

    I2G = imread('images/pier.jpg')
    I2 = imread_gray('images/pier.jpg')

    fig.add_subplot(2,2, 2, title='pier.jpg')

    #FC = findedges(I2, 0.75, 0.20)
    #C = hough_find_lines(FC, 300, 300, 0)
    C = hough_find_lines(I2, 300, 300, 0.16)
    plt.imshow(C)
    
    fig.add_subplot(2,2, 3)

    zadnja(B, 300, 300, I1G, n_najvecjih(15, B))

    fig.add_subplot(2,2, 4)

    zadnja(C, 300, 300, I2G, n_najvecjih(15, C))
    
    plt.show()
    return 0

def zadnja(I, theta_bins, rho_bins, origi, RezSet):
    G = origi.copy()
    diagonala = np.sqrt(origi.shape[0]**2 + origi.shape[1]**2)
    I = nonmaxima_supression_box(I)

    thetaA = np.linspace(- np.pi / 2, np.pi / 2, theta_bins)

    for i in RezSet:
        x = i[0]
        y = i[1]
        rho = ((2 * diagonala * x) / rho_bins) - diagonala
        draw_line(rho, thetaA[y], G.shape[0], G.shape[1])
    
    plt.imshow(G, cmap='gray')

    return 0    

def n_najvecjih(n, I1):
    I = I1.copy()
    sez = []
    for i in range(0, n):
        idx = np.unravel_index(np.argmax(I), I.shape)
        sez.append(idx)
        I[idx] = 0
    return sez

def circle_h():
    radij = 50
    I = imread_gray('images/eclipse.jpg')
    X = non_maxima(I, 1)
    Rez = np.zeros((X.shape))
    X = np.where(X > 0.08, 1, 0)
    X = np.where(X > 0)
    mag = gradient_magnitude(I, 1)

    Prva = X[0]
    Druga = X[1]

    for i in range(0, len(Prva)):
        x = Prva[i]
        y = Druga[i]
        kot = mag[1][x,y]
        a = int(x - radij * np.cos(kot))
        b = int(y + radij * np.sin(kot))

        if 0 <= a and a < Rez.shape[0] and 0 <= b and b < Rez.shape[1]:
            Rez[a,b] +=1

    plt.imshow(Rez)
    plt.show()
    return 0

#prva()
#druga()
tretja()
#cetrta()
#peta()
#circle_h()