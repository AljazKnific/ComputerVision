from numpy import array, int8, uint8
import numpy
from UZ_utils import *
from a2_utils import *

## Že narejena naloga, kjer moramo upoštevati robove.
## Način kateri je bil uporabljen: N točk podvoji na obeh straneh arraya.
## Leva stran -> prvih N točk, Desna stran -> zadnjih N točk
def simple_convolution(I, kernal):
    I2 = np.copy(I)
    rez = []
    kernal = kernal[::-1]
    dol = int((len(kernal) - 1) / 2)

    ### dodatek za c nalogo, kjer podvojimo pixle na robovih
    #"""
    first = I2[0:dol]
    last = I2[len(I2)-dol:len(I2)]

    I2 = np.concatenate((I2, last))
    I2 = np.concatenate((first, I2))
    #"""

    for i in range(0, (len(I2) - len(kernal) + 1)):
        x = I2[i:(i+len(kernal))]
        x = x * kernal
        rez.append(np.sum(x))

    """
    
    img = cv2.filter2D(src=I, ddepth= -1, kernel=kernal)

    plt.plot(I, color='b', label='Original')
    plt.plot(kernal, color='y', label='Kernel')
    plt.plot(img, color='r', label='cv2')
    plt.plot(rez, color='g', label='Result')

    plt.legend()
    plt.show()
    """

    return rez

def gaus_ker(sigma):
    vel = int((np.ceil(sigma * 3) * 2) + 1)
    kernal = []
    
    gor = int(np.ceil(vel/2))
    dol = int(-np.floor(vel/2))

    for x in range(dol, gor):
        rez = (1 / (np.sqrt(2 * np.pi) * sigma)) 
        e = np.exp(-((x**2)/(2 * sigma**2)))
        funk = rez * e
        kernal.append(funk)

    kernal = kernal / np.sum(kernal)
    return kernal

def prikaz_gaus():
    G1 = gaus_ker(0.5)
    G2 = gaus_ker(1)    
    G3 = gaus_ker(2)
    G4 = gaus_ker(3)
    G5 = gaus_ker(4)

    plt.plot(range(int(-len(G1) / 2), int(len(G1) / 2)+1),G1, color='b', label='sigma = 0.5')
    plt.plot(range(int(-len(G2) / 2), int(len(G2) / 2)+1),G2, color='y', label='sigma = 1')
    plt.plot(range(int(-len(G3) / 2), int(len(G3) / 2)+1),G3, color='g', label='sigma = 2')
    plt.plot(range(int(-len(G4) / 2), int(len(G4) / 2)+1),G4, color='r', label='sigma = 3')
    plt.plot(range(int(-len(G5) / 2), int(len(G5) / 2)+1),G5, color='m', label='sigma = 4')

    plt.legend()
    plt.show()

    return 0

def asociativnost():
    k1 = gaus_ker(2)
    k2 = [0.1, 0.6, 0.4]
    I = read_data('signal.txt')


    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(1, 4, 1)
    plt.plot(I)
    plt.title('signal')

    prva = simple_convolution(I, k1)
    prva = simple_convolution(prva, k2)

    fig.add_subplot(1, 4, 2)
    plt.plot(prva)
    plt.title('(signal * k1) * k2')

    druga = simple_convolution(I, k2)
    druga = simple_convolution(druga, k1)

    fig.add_subplot(1, 4, 3)
    plt.plot(druga)
    plt.title('(signal * k2) * k1')

    tretja = simple_convolution(k1, k2)
    tretja = simple_convolution(I, tretja)

    fig.add_subplot(1, 4, 4)
    plt.plot(tretja)
    plt.title('signal * (k2 * k1)')

    plt.show()

    return 0

##Klicanje funkcij

## B) in C) 
I = read_data('signal.txt')
ker = read_data('kernel.txt')
#simple_convolution(I, ker)

## D) , prikaz različnih gausovih kernelov
#prikaz_gaus()

## E)
#asociativnost()


##THIRD QUESTION
#Can you recognize the shape of the kernel? What is the sum of the
#elements in the kernel? How does the kernel affect the signal?

# Gaussian kernel
#Seštevek elementov v kernelu je okoli 1.
# Kernel gladka signal. Gaussov kernel z veliko varianco uteži pixle, ko si bolj na koncih. Gaussov kernel z majhno varianco uteži pixle, ki so bolj na sredini.