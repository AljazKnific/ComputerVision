from UZ_utils import *
from a4_utils import *
import numpy as np
from ex1 import harris_point_det, hessian_points
import cv2

def find_correspondences(desc1, desc2, points1, points2):

    out = np.zeros_like(points1)

    indexi = []

    for i in range(0, desc1.shape[0]):
        minim = 1000
        vrednost = 1000000
        for j in range(0, desc2.shape[0]):
            x = desc1[i]
            s = desc2[j]
            rez = np.sum(np.subtract(np.sqrt(s), np.sqrt(x))**2)
            rez = rez * 0.5
            rez = np.sqrt(rez)
            if vrednost > rez:
                vrednost = rez
                minim = j

        out[i] = points2[minim]
        indexi.append(minim)
    


    return out, np.array(indexi)


def find_matches(I_1, I_2):
    ##
    Harris1 = harris_point_det(I_1, 3, 0.000001, 0.06)   
    Harris2 = harris_point_det(I_2, 3, 0.000001, 0.06)

    desc1 = simple_descriptors(I_1, Harris1[1][:,0], Harris1[1][:,1])
    desc2 = simple_descriptors(I_2, Harris2[1][:,0], Harris2[1][:,1])    

    
    points1 = find_correspondences(desc1, desc2, Harris1[1], Harris2[1])
    points2 = find_correspondences(desc2, desc1, Harris2[1], Harris1[1])
    """
    ##

    I1 = hessian_points(I_1, 3, 0.004)
    I2 = hessian_points(I_2, 3, 0.004)

    desc1 = simple_descriptors(I_1, I1[1][:,0], I1[1][:,1])
    desc2 = simple_descriptors(I_2, I2[1][:,0], I2[1][:,1]) 

    points1 = find_correspondences(desc1, desc2, I1[1], I2[1])
    points2 = find_correspondences(desc2, desc1, I2[1], I1[1])
    """

    sez1 = []
    sez2 = []

    for i in points1[1]:
        ind2 = points2[1][i]
        if i == points1[1][ind2]:
            sez2.append(points1[0][ind2])
            sez1.append(points2[0][i])

    return np.array(sez1), np.array(sez2)

## ni popolno, mogoče popravi kodo, mogoče popravi parametre
def prva():
    I_1 = imread_gray('data/graf/graf_a_small.jpg')
    I_2 = imread_gray('data/graf/graf_b_small.jpg')

    Harris1 = harris_point_det(I_1, 3, 0.000001, 0.06)   
    Harris2 = harris_point_det(I_2, 3, 0.000001, 0.06)
    desc1 = simple_descriptors(I_1, Harris1[1][:,0], Harris1[1][:,1])
    desc2 = simple_descriptors(I_2, Harris2[1][:,0], Harris2[1][:,1])    
    points2 = find_correspondences(desc1, desc2, Harris1[1], Harris2[1])

    display_matches(I_1, Harris1[1], I_2, points2[0])
    """
    I1 = hessian_points(I_1, 3, 0.004)
    I2 = hessian_points(I_2, 3, 0.004)

    desc1 = simple_descriptors(I_1, I1[1][:,0], I1[1][:,1])
    desc2 = simple_descriptors(I_2, I2[1][:,0], I2[1][:,1])
    print(I1[1].shape)
    print(desc1.shape) 

    points2 = find_correspondences(desc1, desc2, I1[1], I2[1])

    display_matches(I_1, I1[1], I_2, points2[0])
    """

    return 0

def druga():
    I_1 = imread_gray('data/graf/graf_a_small.jpg')
    I_2 = imread_gray('data/graf/graf_b_small.jpg')
    sez1, sez2 = find_matches(I_1, I_2)

    display_matches(I_1, sez1, I_2, sez2)
    return 0

## prikaz keypointsov z metodo SIFT
def video():
    vidcap = cv2.VideoCapture('data/Video_UZ.mp4')
    success,image = vidcap.read()

    count = 0
    sift = cv2.SIFT_create(edgeThreshold=60, sigma=0.5, nfeatures=100)
    while success:   
        success,image = vidcap.read()

        if success == 0:
            break

        name = 'frame' + str(count) + '.jpg'
        print("FRAME " + name)

        kp = sift.detect(image,None)
        out = cv2.drawKeypoints(image, kp, image)

        cv2.imshow('Video',out)
        # ESC tipka konča video
        if cv2.waitKey(5) & 0xFF == 27:
            break
        count += 1
    vidcap.release()
    cv2.destroyAllWindows()

    return 0

#prva()
#druga()
#video()

## Questions

# What do you notice when visualizing the correspondences? How ac-curate are the matches?
# Lahko opazimo, da so povezana točke istih delov slik. Nekatere točke so napačno povezane, saj jih 
# morda ni prisotnih na drugi sliki.