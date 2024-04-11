from a6_utils import *
import numpy as np
from UZ_utils import *
import os
from numpy.linalg import inv, det

def dataPrep(st):
    sez = []
    for x in os.listdir('data/faces/' + st):
        I = imread_gray('data/faces/'+ st + '/'+ x)
        sez.append(I.flatten())
        
    sez = np.array(sez)
    return sez

def dualPCA(X):
    
    N = X.shape[1]
    odstej = np.zeros((X.shape[0], 1))
    #mean value
    for i in range(0,X.shape[0]):
        odstej[i] = np.mean(X[i,:])
    
    #center the data
    
    X_d = X - odstej
    #covariance matrix
    C = np.dot(X_d.T, X_d)
    C = C * (1 / (N - 1))
    
    U, S, VT = np.linalg.svd(C)
    L = np.diag(np.sqrt(1 / (S * (N - 1))))
    U = X_d @ U @ L
    
    return odstej,U.T

def numberOfComp(X, mu, U):
    num = 32
    fig = plt.figure(figsize=(5, 5))
    L = np.copy(X)

    for i in range(0,6):
        fig.add_subplot(1, 6, i + 1)
        y = np.dot(U, (L.reshape(-1,1) - mu))

        y[num:] = 0

        x_q = np.dot(U.T, y) + mu
        plt.imshow(x_q.reshape((96,84)), cmap='gray')
        plt.title(str(num))
        num = int(num / 2)
    
    plt.show()
    return 0

def eDel(U, mu):
    I =  imread_gray('data/elephant.jpg')
    E  = I.flatten()
    y1 = np.dot(U, (I.reshape(-1,1) - mu))
    x_q1 = np.dot(U.T, y1) + mu
    
    fig = plt.figure(figsize=(5, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title("Original elephant.jpg")
    fig.add_subplot(1, 2, 2)
    plt.imshow(x_q1.reshape((96,84)), cmap='gray')
    plt.title("Transformed elephant")

    plt.show()
    
    return 0

def animacija(I, U, mu, fir, sec):
    y = np.dot(U, (I.reshape(-1,1) - mu))
    scal = 3000
    lin = np.linspace(-10, 10, 100)
    sR = np.sin(lin) * scal
    cR = np.cos(lin) * scal


    for k, j in zip(sR, cR):
        y[fir] = k
        y[sec] = j
        x_q = np.dot(U.T, y) + mu
        plt.imshow(x_q.reshape((96,84)), cmap='gray')
        plt.draw()
        plt.pause(.00001)

    plt.close()
    return 0

# c - število razredov
# n - število primerov na razred
# X - primerki razredov, razporejeni po stolpcih
def LDA(X, c, n):
    mean_overall = np.mean(X, axis=1, keepdims=True)

    SB = np.zeros((X.shape[0], X.shape[0]))
    SW = np.zeros((X.shape[0], X.shape[0]))
    mean_class = np.zeros((X.shape[0], c))

    for i in range(c):
        mean_class[:,i] = np.mean(X[:, n * i: (i + 1) * n], axis=1)

        SB += n * np.dot((mean_class[:, [i]] - mean_overall),  (mean_class[:, [i]] - mean_overall).T)

        for j in range(n):
            SW += np.dot(X[:, [i * n + j]] - mean_class[:, [i]], (X[:, [i * n + j]] - mean_class[:, [i]]).T)

    
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
    eigenvectors = eigenvectors.T 
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]

    return eigenvectors, mean_overall



def prva():
    # izberi med 1, 2, 3
    X = dataPrep('1')

    #b del naloge
    mu, U = dualPCA(X.T)
    #prvi izris
    #Using dual PCA
    """
    fig = plt.figure(figsize=(5, 5))

    for i in range(0,5):
        T = U.T[:,i].reshape((96,84))
        fig.add_subplot(1, 5, i + 1)
        plt.imshow(T, cmap='gray')

    plt.show()
    """

    #Eigenvectors z največjimi lastnimi vrednostmi so tisti, ki zajemajo največjo raznolikost v podatkih 
    #in so zato najpomembnejši za identifikacijo vzorcev in trendov v podatkih.
    """
    fig = plt.figure(figsize=(5, 5))
    #prikaz prve slike transformirane iz Pca v image space
    fig.add_subplot(1, 5, 1)

    firImage = np.copy(X[0])

    y1 = np.dot(U, (firImage.reshape(-1,1) - mu))
    x_q1 = np.dot(U.T, y1) + mu
    B = x_q1.reshape((96,84))
    plt.title("PCA -> Image")
    plt.imshow(B, cmap='gray')

    # izrisi B z dodatkom crne pike
    fig.add_subplot(1, 5, 2)
    B2 = np.copy(firImage)
    B2[4074] = 0
    plt.title("Image space I[4074] = 0")
    plt.imshow(B2.reshape((96,84)), cmap='gray')

    # izbrisan vector v PCA spacu
    fig.add_subplot(1, 5, 3)
    y2 = np.dot(U, (B2.reshape(-1,1) - mu))
    x_q2 = np.dot(U.T, y2) + mu
    C = x_q2.reshape((96,84))
    plt.title("I[4074] = 0 -> PCA -> Image")
    plt.imshow(C, cmap='gray')

    #razlika med prvo in 3 sliko
    fig.add_subplot(1, 5, 4)
    plt.title("Razlika med 3 in 1")
    plt.imshow(np.subtract(C,B), cmap='gray')

    #prikaz izbirsa 4 komponente v PCA prostoru
    fig.add_subplot(1, 5, 5)

    B3 = np.copy(firImage)
    y3 = np.dot(U, (B3.reshape(-1,1) - mu))
    y3[4] = 0
    x_q3 = np.dot(U.T, y3) + mu
    C2 = x_q3.reshape((96,84))
    plt.title("I -> PCA (i[4] = 0) -> Image space")
    plt.imshow(C2, cmap='gray')

    plt.show()
    """

    # Effect of the number of components on the reconstruciton
    """
    secImage = np.copy(X[0])
    numberOfComp(secImage, mu, U)
    """

    #Če se ohrani veliko število komponent, bo obnova izvirnih podatkov bolj natančna, vendar bo zmanjšanje dimenzionalnosti manj pomembno. Nasprotno, če se ohrani manjše število komponent,
    # bo obnova izvirnih podatkov manj natančna, vendar bo zmanjšanje dimenzionalnosti bolj pomembno.

    #Informativeness of each component
    
    X2 = dataPrep('2')
    mu2, U2 = dualPCA(X2.T)
    avI = np.mean(X2, axis=0)
    animacija(avI, U2, mu2, 0, 1)
    

    # Reconstruction of a foreign image
    #Učinkovitost transformacije v PCA prostor je odvisna od podobnosti med sliko in učnimi slikami, ki so bile uporabljene za izgradnjo PCA prostora. Če je slika zelo drugačna od učnih slik, transformacija morda ne bo zmožna natančno predstaviti značilnosti slike, 
    #kar bo povzročilo slabšo rekonstrukcijo.
    #eDel(U, mu)

    #LDA implementation
    #"""
    fig = plt.figure(figsize=(5, 5))
    X2 = dataPrep('2')
    X3 = dataPrep('3')
    T = np.concatenate((X, X2, X3))
    #tocke vseh setov v pca
    mu2, U2 = dualPCA(T.T)
    y = np.dot(U2, (T.T - mu2))
    fig.add_subplot(1, 2, 1)
    y = y * -1
    plt.scatter(y[0], y[1], c=['purple']*64 + ['green']*64 + ['yellow']*64)
    #LDA
    y = y * -1
    eigenvectors, mean_overall = LDA(y[:30], 3, 64)

    x_q = np.dot(eigenvectors, y[:30] - mean_overall)
    fig.add_subplot(1, 2, 2)
    x_q *= -1
    print(x_q.shape)
    plt.scatter(x_q[1], x_q[0], c=['purple']*64 + ['green']*64 + ['yellow']*64)
    plt.show()
    #"""
    
    
    return 0

prva()