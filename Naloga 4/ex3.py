from UZ_utils import *
from a4_utils import *
import numpy as np
import random
from ex2 import find_matches

## izračun evklidske razdalje
def euclidean_distance(point1, point2, h):

        temp = np.zeros((3,1))
        temp[0] = point1[1]
        temp[1] = point1[0]
        temp[2] = 1
        a = h.dot(temp)
        a = a / a[-1]

        temp2 = np.zeros((3,1))
        temp2[0] = point2[1]
        temp2[1] = point2[0]
        temp2[2] = 1

        dist = np.linalg.norm(temp2 - a)

        return dist
## ocena homografije
def estimate_homography(points1, points2):
    A = []

    for i in range(0, points1.shape[0]):
        x_r = points1[i][1]
        y_r = points1[i][0]
        x_t = points2[i][1]
        y_t = points2[i][0]
        A.append([x_r, y_r, 1, 0, 0, 0, - x_t * x_r, -x_t * y_r, -x_t])
        A.append([0, 0, 0, x_r, y_r, 1, -y_t * x_r, -y_t * y_r, -y_t])

    A = np.array(A)
    U, S, VT = np.linalg.svd(A)

    h = VT[-1] / VT[-1][-1]
    h = h.reshape(3,3)
    return h


## pridobi naključne točke in jih izlušči
def getRandomPoints(sez1, sez2):
    A1 = []
    A2 = []
    B = random.sample(range(0, sez1.shape[0]), 4)

    B = set(B)

    for i in B:
        A1.append(sez1[i])
        A2.append(sez2[i])
    
    return np.array(A1), np.array(A2), B

##iskanje inlierjev
def find_inliners(h,threshold, sez1, sez2):
    A = []
    C = []

    errorEuc = 0

    for i in range(0, sez1.shape[0]):

        val = euclidean_distance(sez1[i], sez2[i], h)
        if  val < threshold:
            A.append(sez1[i])
            C.append(sez2[i])
        errorEuc += val

    errorEuc = errorEuc / sez1.shape[0]

    return errorEuc, np.array(A), np.array(C)

##metoda ransac
def RANSACC(sez1, sez2, k, thres):
    
    fail = 1000000
    Rez1 = []
    Rez2 = []
    RezH = []

    for i in range(k):
        A1, A2, B = getRandomPoints(sez1, sez2)
        h = estimate_homography(A1, A2)
        
        errorEuc, prva, druga = find_inliners(h, thres,sez1, sez2)

        if prva.shape[0] >= 4:
            t = estimate_homography(prva, druga)
            errorEuc1, fir, sec = find_inliners(t, thres,sez1,sez2)
            
            if errorEuc1 < fail:
                fail = errorEuc1
                Rez1 = fir
                Rez2 = sec
                RezH = t

    return Rez1, Rez2, RezH

def prva():
    IA = imread_gray('data/newyork/newyork_a.jpg')
    IB = imread_gray('data/newyork/newyork_b.jpg')
    tocke = np.loadtxt('data/newyork/newyork.txt')

    points1 = np.zeros((4,2))
    points2 = np.zeros((4,2))
    points1[:,0] = tocke[:,1]
    points1[:,1] = tocke[:,0]
    points2[:,0] = tocke[:,3]
    points2[:,1] = tocke[:,2]

    display_matches(IA, points1, IB, points2)

    return 0

def druga():
    IA = imread_gray('data/newyork/newyork_a.jpg')
    IB = imread_gray('data/newyork/newyork_b.jpg')
    tocke = np.loadtxt('data/newyork/newyork.txt')

    points1 = np.zeros((4,2))
    points2 = np.zeros((4,2))
    points1[:,0] = tocke[:,1]
    points1[:,1] = tocke[:,0]
    points2[:,0] = tocke[:,3]
    points2[:,1] = tocke[:,2]

    h = estimate_homography(points1, points2)
    first = cv2.warpPerspective(IA, h, IA.shape[::-1])
    plt.imshow(first, cmap='gray')
    plt.show()

    return 0

def tretja():
    IA = imread_gray('data/newyork/newyork_a.jpg')
    IB = imread_gray('data/newyork/newyork_b.jpg')

    #IA = imread_gray('data/graf/graf_a_small.jpg')
    #IB = imread_gray('data/graf/graf_b_small.jpg')

    sez1, sez2 = find_matches(IA, IB)

    x , y, H = RANSACC(sez1, sez2,100,2)
    #display_matches(IA, x, IB, y)

    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(1, 2, 1)
    plt.imshow(IB, cmap='gray')
    plt.title('newyork_b')


    fig.add_subplot(1, 2, 2)
    H = cv2.warpPerspective(IA, H, IA.shape[::-1])
    plt.imshow(H, cmap='gray')
    plt.title('transformed newyork_a')
    
    plt.show()
    return 0

#prva()
#druga()
tretja()

##Question : Looking at the equation above,
#  which parameters account for translation and which for rotation and scale?
# Translation: p3 and p4 Rotation: p1 and p2

##Question: Write down a sketch of an algorithm to determine similarity transformfrom a set of point correspondences
# P= [(xr1,xt1),(xr2,xt2),...(xrn,xtn)]. For moredetails consult the lecture notes.

## Question: How many iterations on average did you need to find a good solution?
# How does the parameter choice for both the keypoint detector and RANSAC itself
# influence the performance (both quality and speed)?

#100 iterations. Več kot je iteracij, dalj časa porabimo za izračun rezultata.
# Vpliv ima tudi sigma. Manjša kot je, dalj časa čakamo na rezultat, saj imamo več zaznanih točk.