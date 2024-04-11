from numpy import array, int8, uint8
import numpy
from UZ_utils import *
from a2_utils import *
from ex2 import gaus_ker

## Kreiranje gaussovega filtra
def gaussfilter(I, sigma):
    kernal = np.array([gaus_ker(sigma)])
    I = cv2.filter2D(src=I, ddepth= -1, kernel=kernal)
    k_transposed = kernal.T
    img = cv2.filter2D(src=I, ddepth= -1, kernel=k_transposed)
    
    return img


def prva():
    I = imread_gray('images/lena.png')
    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2, 3, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original')

    fig.add_subplot(2, 3, 2)
    gau = gauss_noise(I)
    plt.imshow((gau * 255).astype(int), cmap='gray', vmax=255, vmin=0)
    plt.title('Gaussian noise')


    fig.add_subplot(2, 3, 3)
    sal = sp_noise(I)
    plt.imshow(sal, cmap='gray')
    plt.title('Salt and Pepper')

    fig.add_subplot(2, 3, 5)
    fil_gau = gaussfilter(gau, 1.2)
    plt.imshow(fil_gau, cmap='gray')
    plt.title('Filtered Gaussian noise')

    fig.add_subplot(2, 3, 6)
    fil_sp = gaussfilter(sal, 1.7)
    plt.imshow(fil_sp, cmap='gray')
    plt.title('Filtered Salt and Pepper')

    plt.show()


    return 0


## Prikaz muzejev
def sharpen():
    I = imread_gray('images/museum.jpg')
    kernel1 = np.array([[0,0,0], [0,2,0], [0,0,0]])
    kernel2 = np.array([[1,1,1], [1,1,1], [1,1,1]])

    kernel2 = kernel2 / 9
    #print(kernel2)

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original')

    fig.add_subplot(1, 2, 2)
    img1 = cv2.filter2D(src=I, ddepth= -1, kernel=kernel1)
    img2 = cv2.filter2D(src=I, ddepth= -1, kernel=kernel2)
    img = img1 - img2
    img = np.where(img < 0, 0, img)
    img = np.where(img > 1, 1, img)

    
    plt.imshow(img, cmap='gray')
    plt.title('Sharpened')

    plt.show()

    return 0

def simple_median(I, w):
    sez = []
    meja = int(w / 2)
    print(meja)
    for i in range(0,len(I)):
        dol = i - meja
        gor = i + meja

        if dol < 0:

            tab = I[0:gor]

        elif gor > len(I):

            tab = I[dol:len(I)]
        else:

            tab = I[dol:gor]


        sez.append(np.median(tab))


    return sez

def prikaz_median():
    sig = np.zeros(40)
    sig[10:20:1] += 1

    cor_signal = sig.copy()
    cor_signal[2] = 3
    cor_signal[5] = 3
    cor_signal[15] = 3
    cor_signal[16] = 0
    cor_signal[27] = 3
    cor_signal[28] = 3
    cor_signal[30] = 3
    cor_signal[38] = 3

    fig = plt.figure(figsize=(3, 3))


    fig.add_subplot(1, 4, 1)
    plt.plot(sig)
    plt.title('Original')

    plt.ylim([0, 5])

    fig.add_subplot(1, 4, 2)
    plt.plot(cor_signal)
    plt.title('Corrupted')

    plt.ylim([0, 5])

    X = gaussfilter(cor_signal, 2)

    fig.add_subplot(1, 4, 3)
    plt.plot(X)
    plt.title('Gauss')

    plt.ylim([0, 5])
    Z = simple_median(cor_signal, 11)

    fig.add_subplot(1, 4, 4)
    plt.plot(Z)
    plt.title('Median')


    plt.ylim([0, 5])
    plt.show()

    return 0

def median2d_filter(I, w):
    meja = int(w / 2)

    X = np.zeros_like(I)

    height, width = I.shape

    #print(height)
    #print(width)

    for i in range(0, height):
        for j in range(0, width):
            x1= j - meja
            x2= j + meja + 1
            y1= i - meja
            y2= i + meja + 1 

            if x1 < 0:
                x1 = 0
            if x2 > width:
                x2 = width
            if y1 < 0:
                y1 = 0
            if y2 > height:
                y2 = height

            #print("X1 " + str(x1) + " X2 " + str(x2) + " Y1 " + str(y1) + " Y2 " + str(y2))


            temp = I[y1:y2,x1:x2]
            #print(temp)
            temp= temp.reshape(-1)

            temp = np.sort(temp)
            middle = int(len(temp) / 2)
            #print(np.median(temp))
            X[i][j] = np.median(temp[middle])
    return X

def prikaz_naloge_d():
    fig = plt.figure(figsize=(5, 5))

    I = imread_gray('images/lena.png')

    fig.add_subplot(2, 4, 1)

    plt.imshow(I, cmap='gray')
    plt.title('Original')

    fig.add_subplot(2, 4, 2)

    gau = gauss_noise(I)
    plt.imshow((gau * 255).astype(int), cmap='gray', vmax=255, vmin=0)
    plt.title('Gaussian noise')


    fig.add_subplot(2, 4, 3)

    fil_gau = gaussfilter(gau, 1.1)
    plt.imshow(fil_gau, cmap='gray')
    plt.title('Gauss filtered')

    fig.add_subplot(2, 4, 4)

    med_fil_gau = median2d_filter(gau, 3)
    plt.imshow(med_fil_gau, cmap='gray')
    plt.title('Median filtered')

    fig.add_subplot(2, 4, 6)

    sp = sp_noise(I)
    plt.imshow(sp, cmap='gray')
    plt.title('Salt and Pepper')

    fig.add_subplot(2, 4, 7)

    sp_fil_gau = gaussfilter(sp, 1.2)
    plt.imshow(sp_fil_gau, cmap='gray')
    plt.title('Gauss filtered')


    fig.add_subplot(2, 4, 8)

    med_fil_sp = median2d_filter(sp, 3)
    plt.imshow(med_fil_sp, cmap='gray')
    plt.title('Median filtered')

    plt.show()

    return 0


def lap_filter(sigma):
    g = gaus_ker(sigma)
    unit = np.zeros((1, len(g)))
    unit[0][int((len(g) - 1) / 2)] = 1
    rez = unit - g
    return rez


def hybrid_image():
    I1 = imread_gray('images/lincoln.jpg')
    I2 = imread_gray('images/obama.jpg')

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(2, 3, 1)

    plt.imshow(I1, cmap='gray')
    plt.title('image 1')

    fig.add_subplot(2, 3, 2)

    plt.imshow(I2, cmap='gray')
    plt.title('image 2')

    fig.add_subplot(2, 3, 4)

    gau_I1 = gaussfilter(I1, 6)

    plt.imshow(gau_I1, cmap='gray')
    plt.title('Gauss')

    fig.add_subplot(2, 3, 5)

    lap_prvi = lap_filter(35)
    lap_I2 = cv2.filter2D(src=I2, ddepth=-1, kernel=lap_prvi)
    lap_I2 = cv2.filter2D(src=lap_I2, ddepth=-1, kernel = lap_prvi.T)


    plt.imshow(lap_I2, cmap='gray')
    plt.title('Laplace')

    result = gau_I1 + lap_I2

    lap_drugi = lap_filter(130)
    result = cv2.filter2D(src=result, ddepth=-1, kernel=lap_drugi)
    result = cv2.filter2D(src=result, ddepth=-1, kernel=lap_drugi.T)


    fig.add_subplot(2, 3, 3)

    plt.imshow(result, cmap='gray')
    plt.title('Result')

    plt.show()

    return 0

## Klici funckij
#A) Naloga
#prva()

#B) naloga
#sharpen()

#C) naloga
#prikaz_median()

#D) naloga
#prikaz_naloge_d()

#E) naloga
#hybrid_image()

##FOURTH QUESTION
#Which noise is better removed using the Gaussian filter?

#Gaussian noise

##FIFTH QUESTION
# Which filter performs better at this specific task? In comparison to
#Gaussian filter that can be applied multiple times in any order, does the order
#matter in case of median filter? What is the name of filters like this?

# V primeru šuma sol in poper je za filtriranje najboljši median filter.
# Vrstni red pri median filtru je pomemben, saj spada med nelinearne filtre.
# Nonlinear filters


##SIXTH QUESTION
# What is the computational complexity of the Gaussian filter operation?
#How about the median filter? What does it depend on? Describe the computational
#complexity using the O(·) notation (you can assume n log n complexity for sorting).

# Časovna zahtevnost =  2D gaussian -> O(n^2 * w * h) n-velikost filtra w-visina slike h-sirina slike
#                       1D gaussian -> O(n * w * h) + O(n * w * h) = O(n * w * h)
#                       Median      -> O(w * h * n^2logn)