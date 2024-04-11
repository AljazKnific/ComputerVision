from operator import attrgetter
from numpy import array, int8, uint8
import numpy
from UZ_utils import *
from a2_utils import *
import os

class Paket:
    def __init__(self, path, distance, histogram, slika):
        self.path = path
        self.distance = distance
        self.histogram = histogram
        self.slika = slika

def preberi(path):
    I = Image.open(path).convert('RGB')  # PIL image.
    I = np.asarray(I)  # Converting to Numpy array.
    return I


## a naloga
## Ustvarim 3d array v katerem inkrementiram vrednosti glede na vrednosti rgb vsakega pixla

def myhist3(n_bins, slika):
    H = np.zeros((n_bins, n_bins, n_bins))
    L = slika.reshape(-1,3)
    
    velikost = 256 / n_bins

    L = np.floor(L / velikost)

    for x in range(0, L.shape[0]):
        r,g,b = L[x]
        H[int(r)][int(g)][int(b)] += 1 

    H = H / np.sum(H)
    return H

##Primerjava histogramov, na različne načine 
def compare_histograms(H1, H2, nacin):
    if nacin == 'L2':
        H = np.subtract(H1, H2)
        H = H * H
        sestej = np.sum(H)
        rez = np.sqrt(sestej)
        return rez

    if nacin == 'CHI':
        H = np.subtract(H1, H2)
        H = H * H
        T = np.add(H1, H2)
        T = T + (1.e-10)
        H = np.divide(H, T)
        rez = np.sum(H)
        rez = rez / 2
        return rez
    
    if nacin == 'INT':
        X = np.minimum(H1, H2)
        rez = np.sum(X)
        rez = 1 - rez
        return rez
    
    if nacin == 'HEL':
        H1 = np.sqrt(H1)
        H2 = np.sqrt(H2)
        H = np.subtract(H1, H2)
        H = H * H
        rez = np.sum(H)
        rez = rez / 2
        rez = np.sqrt(rez)
        return rez
    return 0

## Prikaz rešitev c naloge
def prva_naloga():
    slika1 = preberi('dataset/object_01_1.png')
    H1 = myhist3(8,slika1)

    slika2 = preberi('dataset/object_02_1.png')
    H2 = myhist3(8,slika2)
    
    slika3 = preberi('dataset/object_03_1.png')
    H3 = myhist3(8,slika3)

    rez1 = compare_histograms(H1, H1, 'L2')
    rez2 = compare_histograms(H1, H2, 'L2')
    rez3 = compare_histograms(H1, H3, 'L2')
    H1 = H1.reshape(-1, order="F")
    H2 = H2.reshape(-1, order="F")
    H3 = H3.reshape(-1, order="F")
    
    print("L2 -> H1 , H2 = " + str(rez2))
    print("L2 -> H1 , H3 = " + str(rez3))
    print("CHI-SQUARE DISTANCE -> H1 , H2 = " + str(compare_histograms(H1, H2, 'CHI')))
    print("CHI-SQUARE DISTANCE -> H1 , H3 = " + str(compare_histograms(H1, H3, 'CHI')))
    print("INTERSECTION -> H1 , H2 = " + str(compare_histograms(H1, H2, 'INT')))
    print("INTERSECTION -> H1 , H3 = " + str(compare_histograms(H1, H3, 'INT')))
    print("HELLINGER DISTANCE -> H1 , H2 = " + str(compare_histograms(H1, H2, 'HEL')))
    print("HELLINGER DISTANCE -> H1 , H3 = " + str(compare_histograms(H1, H3, 'HEL')))
    
    fig = plt.figure(figsize=(4, 4))

    fig.add_subplot(2, 3, 1)
    plt.imshow(slika1)
    plt.title('Image 1')

    fig.add_subplot(2, 3, 2)
    plt.imshow(slika2)
    plt.title('Image 2')

    fig.add_subplot(2, 3, 3)
    plt.imshow(slika3)
    plt.title('Image 3')

    fig.add_subplot(2, 3, 4)
    plt.bar(range(512), H1, width=5)
    plt.title('L2(H1, H1) = ' + str(round(rez1, 2)))

    fig.add_subplot(2, 3, 5)
    plt.bar(range(512), H2, width=5)
    plt.title('L2(H1, H2) = ' + str(round(rez2, 2)))

    fig.add_subplot(2, 3, 6)
    plt.bar(range(512), H3, width=5)
    plt.title('L2(H1, H3) = ' + str(round(rez3, 2)))

    plt.show()
    
    return 0
## Preberem vse slike iz direktorija
## Vsaki sliki dodelim histogram, ustrezno primerjam histograma z razdaljo
## Vse slike uredim po razdalji, ter prikažem 6 najbližjih
def druga_naloga(n_bins, path, st, nacin):
    sez = []
    for x in os.listdir(path): 
        sez.append(path + x)

    name = sez[st]
    I = preberi(name)
    H = myhist3(n_bins, I)
    izbor = []

    for y in sez:
        I1 = preberi(y)
        H1 = myhist3(n_bins, I1)
        razd = compare_histograms(H, H1, nacin)
        H1 = H1.reshape(-1)
        p = Paket(y, razd, H1, I1)
        izbor.append(p)

    vrni = izbor
    
    #"""
    izbor.sort(key=attrgetter('distance'))
    fig = plt.figure(figsize=(5,5))
    for i in range(0,6):
        fig.add_subplot(2, 6, i + 1)
        plt.imshow(izbor[i].slika)
        plt.title(izbor[i].path)
    
        fig.add_subplot(2, 6, i + 7)
        plt.bar(range(n_bins**3), izbor[i].histogram, width=5)
        plt.title(nacin + " = " + str(round(izbor[i].distance, 2)))

    plt.show()
    #"""
    return vrni

## Prikaz enakih točk na urejenem in neurejenem grafu
def tretja_naloga():
    nabor = druga_naloga(8, 'dataset/', 19, "CHI")

    unsorted = [o.distance for o in nabor]
    sorted_ = np.sort(unsorted)

    points = np.where(unsorted < sorted_[5], True, False)

    fig = plt.figure(figsize=(5,5))
    fig.add_subplot(1, 2, 1)
    plt.plot(unsorted, markevery=points, marker='o', fillstyle='none', mec='red')

    fig.add_subplot(1, 2, 2)
    plt.plot(sorted_, markevery=np.where(sorted_ < sorted_[5], True, False), marker='o', fillstyle='none', mec='red')
    plt.show()
    return 0


## Klicanje funkcij prve naloge

prva_naloga()
#druga_naloga(8, 'dataset/', 19, 'HEL')
#tretja_naloga()

###FIRST QUESTION: 
#Which image (object_02_1.png or object_03_1.png) is more similar
#to image object_01_1.png considering the L2 distance? How about the other three
#distances? We can see that all three histograms contain a strongly expressed component (one bin has a much higher value than the others). 
# Which color does this bin represent?

#Najbolj podoben objektu 1 je objekt 3. To lahko vidimo po razdaljah, ki prikazujejo razlike med histogramoma. Manjša kot je razdalja bolj sta si histograma podobna.


###SECOND QUESTION
#Which distance is in your opinion best suited for image retrieval? How
#does the retrieved sequence change if you use a different number of bins? Is the
#execution time affected by the number of bins?

#Menim, da je najbolsi nacin Hellingerjev, saj se mi funckija za izračun zdi najbolj kompleksna izmed naštetih.
#S povečanjem števila binov, dobimo pri iskanju druge slike.(8 bins nam vrne drugacne slike kot 16 binsov)
#Več kot je binov -> večji je čas izvedbe.