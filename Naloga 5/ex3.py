from UZ_utils import *
from a5_utils import *
from matplotlib import *

def triangulation(points1, points2, P1, P2):
    rez = []
    for i in range(0, points1.shape[0]):
        ax = points1[i][0]
        ay = points1[i][1]
        az = 1
        x1 = np.array([ [0,-az,ay],
                        [az,0,-ax],
                        [-ay,ax,0]])        
        
        bx = points2[i][0]
        by = points2[i][1]
        bz = 1
        x2 = np.array([ [0,-bz,by],
                        [bz,0,-bx],
                        [-by,bx,0]])
        
        X_2 = x2 @ P2
        X_1 = x1 @ P1
        
        A = [X_1[0], X_1[1], X_2[0], X_2[1]]
        
        U,S,V = np.linalg.svd(A)
        
        z = V[-1] / V[-1][-1]
        
        z = z[:3]
        rez.append(z)
        
    return rez

def prva():
    T = np.array([[-1,0,0],
                  [0,0,1],
                  [0,-1,0]])
    t1 = np.loadtxt('data/epipolar/house1_camera.txt')
    t2 = np.loadtxt('data/epipolar/house2_camera.txt')
    I1 = imread_gray('data/epipolar/house1.jpg')
    I2 = imread_gray('data/epipolar/house2.jpg')
    
    tocke = np.loadtxt('data/epipolar/house_points.txt')
    points1 = np.zeros((10,2))
    points2 = np.zeros((10,2))
    
    points1[:,0] = tocke[:,0]
    points1[:,1] = tocke[:,1]
    points2[:,0] = tocke[:,2]
    points2[:,1] = tocke[:,3]
    
    rez = triangulation(points1, points2, t1, t2)
    rezFinal = np.dot(rez, T)
    
    fig = plt.figure(figsize=(5, 5))

    fig.add_subplot(1, 2, 1)
    plt.imshow(I1, cmap='gray')
    plt.plot(points1[:,0], points1[:,1], 'bo')
    
    fig.add_subplot(1, 2, 2)
    plt.imshow(I2, cmap='gray')
    plt.plot(points2[:,0], points2[:,1], 'bo')

    plt.show()
    
    ax = plt.axes(projection='3d')
    for i,m in enumerate(rezFinal):
        ax.scatter([m[0]], [m[1]], [m[2]], color='red')
        ax.text(m[0], m[1], m[2], str(i))
    plt.show()
    
    
    return 0

prva()