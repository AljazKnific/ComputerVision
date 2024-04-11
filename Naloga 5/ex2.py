from UZ_utils import *
from a5_utils import *

def fundamental_matrix(points1, points2):
    A = []
    points1,T1 = normalize_points(points1)
    points2, T2 = normalize_points(points2)
    

    for i in range(0, points1.shape[0]):
        u = points1[i][0]
        v = points1[i][1]
        u_ = points2[i][0]
        v_ = points2[i][1]
        A.append([u * u_, u_ * v, u_, u * v_, v * v_, v_, u, v, 1])

    A = np.array(A)
    U, S, VT = np.linalg.svd(A)

    F = VT[-1]
    F = F.reshape(3,3)

    U, D , F2 = np.linalg.svd(F)
    #lowest eigen value to zero
    D[-1] = 0
    
    F = ( U * D) @ F2
    
    F = T2.T @ F @ T1

    return F


def prva():
    tocke = np.loadtxt('data/epipolar/house_points.txt')
    I1 = imread_gray('data/epipolar/house1.jpg')
    I2 = imread_gray('data/epipolar/house2.jpg')

    points1 = np.zeros((10,2))
    points2 = np.zeros((10,2))
    
    points1[:,0] = tocke[:,0]
    points1[:,1] = tocke[:,1]
    points2[:,0] = tocke[:,2]
    points2[:,1] = tocke[:,3]
    F = fundamental_matrix(points1, points2)
    
    p1 = np.zeros((1,2))
    p2 = np.zeros((1,2))
    p1[0][0] = 85
    p1[0][1] = 233
    p2[0][0] = 67
    p2[0][1] = 219
    Fir = reprojection_error(F, p1, p2)
    
    print(reprojection_error(F, points1, points2))
    
    
    plt.imshow(I1, cmap='gray')
    plt.plot(points1[:,0], points1[:,1], 'bo', color='r')

    for i in range(0,10):
        temp2 = np.zeros((3,1))
        temp2[0] = points2[i][0]
        temp2[1] = points2[i][1]
        temp2[2] = 1
        
        #print(temp2)

        l = F.T @ temp2

        draw_epiline(l, I1.shape[0], I1.shape[1])
        
    plt.show()
        
        
    plt.imshow(I2, cmap='gray')
    plt.plot(points2[:,0], points2[:,1], 'bo', color='r')
    for i in range(0,10):
        temp = np.zeros((3,1))
        temp[0] = points1[i][0]
        temp[1] = points1[i][1]
        temp[2] = 1    
        
        l_ = F @ temp
        
        draw_epiline(l_, I2.shape[0], I2.shape[1])
    
    plt.show()
    
    return F

def reprojection_error(F, points1, points2):
    dis = 0
    for i in range(0,points1.shape[0]):
        temp = np.zeros((3,1))
        temp[0] = points1[i][0]
        temp[1] = points1[i][1]
        temp[2] = 1 
        
        temp2 = np.zeros((3,1))
        temp2[0] = points2[i][0]
        temp2[1] = points2[i][1]
        temp2[2] = 1   
        
        l_ = F @ temp
        l = F.T @ temp2
        
        
        dis1 = np.abs(l[0] * temp[0] + l[1] * temp[1] + l[2]) / np.sqrt(l[0]**2 + l[1]**2)
        dis2 = np.abs(l_[0] * temp2[0] + l_[1] * temp2[1] + l_[2]) / np.sqrt(l_[0]**2 + l_[1]**2)
        dis += ((dis1 + dis2) / 2)
    
    dis = dis/ points1.shape[0]
    return dis

prva()