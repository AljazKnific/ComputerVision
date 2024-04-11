from a6_utils import *
import numpy as np


def closestPoint(t, q, F):
    min = 1000
    index = -1
    x = t - q
    x = x**2
    N = x.shape[1]
    if F == 'T':
        N = N - 1
    for i in range(0, N):
        d = np.sqrt(x[0,i] + x[1,i])
        if d < min: 
            index = i
            min = d
    return index

def PCA(X):
    N = X.shape[1]
    odstej = np.zeros((X.shape[0], 1))
    #mean value
    for i in range(0,X.shape[0]):
        odstej[i] = np.mean(X[i,:])
    
    #center the data
    
    X_d = X - odstej
    #covariance matrix
    C = np.dot(X_d, X_d.T)
    C = C * (1 / (N - 1))
    
    U, S, VT = np.linalg.svd(C)
    return odstej, C, U, S, VT 

def prva():
    tocke = np.loadtxt('data/points.txt')
    
    points = np.zeros((5,2))
    
    points[:,0] = tocke[:,0]
    points[:,1] = tocke[:,1]
    mu, C, U, S, VT = PCA(points.T)
    # prikaz b in c naloge
    """
    drawEllipse(mu, C)
    plt.scatter(points[:,0], points[:,1], marker='+')
    for i in range(points.shape[0]):
        plt.text(points[i,0], points[i,1], str(i + 1))
    
    ##dodajanje c naloge
    for x in range(0, U.shape[0]):
        plt.plot([mu[0], mu[0] + U[x,0] * np.sqrt(S[x])], [mu[1], mu[1] + U[x,1] * np.sqrt(S[x])])
    plt.xlim([-1, 7])
    plt.ylim([-1, 7])
    plt.show()
    """
    """
    # izgubili bi okoli 80% podatkov
    # d naloga
    cdf = np.cumsum(S)
    cdf = cdf / np.max(cdf)

    plt.bar(range(len(cdf)), cdf)
    plt.show()
    """
    
    # naloga e
    """
    UT = np.copy(U)
    
    y = np.dot(UT.T,(points.T - mu))
    
    UT[1,:] = 0
    
    x_q = np.dot(UT.T,y) + mu
    
    drawEllipse(mu, C)
    plt.scatter(points[:,0], points[:,1], marker='+')
    for i in range(points.shape[0]):
        plt.text(points[i,0], points[i,1], str(i + 1))

    for x in range(0, U.shape[0]):
        plt.plot([mu[0], mu[0] + U[x,0] * np.sqrt(S[x])], [mu[1], mu[1] + U[x,1] * np.sqrt(S[x])])
        
        plt.scatter(x_q.T[:,0], x_q.T[:,1])
    for i in range(x_q.T.shape[0]):
        plt.text(x_q.T[i,0], x_q.T[i,1], str(i + 1))
    plt.show()
    """
    
    # naloga f
    """
    q_point = np.array([[6],[6]])
    cPoint = closestPoint(points.T, q_point, "")
    print("First calculation closest point= " + str(cPoint + 1))
    
    points2 = np.zeros((6,2))
    for j in range(0, tocke.shape[0]):
        points2[j,0] = tocke[j,0]
        points2[j,1] = tocke[j,1]
    points2[5,0] = 6
    points2[5,1] = 6
    
    UT = np.copy(U)
    
    y2 = np.dot(UT.T,(points2.T - mu))
    
    UT[1,:] = 0
    
    x_q2 = np.dot(UT.T,y2) + mu
    
    drawEllipse(mu, C)
    plt.scatter(points2[:,0], points2[:,1], marker='+')
    for i in range(points2.shape[0]):
        plt.text(points2[i,0], points2[i,1], str(i + 1))

    for x in range(0, U.shape[0]):
        plt.plot([mu[0], mu[0] + U[x,0] * np.sqrt(S[x])], [mu[1], mu[1] + U[x,1] * np.sqrt(S[x])])
        
        plt.scatter(x_q2.T[:,0], x_q2.T[:,1])
    for i in range(x_q2.T.shape[0]):
        plt.text(x_q2.T[i,0], x_q2.T[i,1], str(i + 1))
    
    xPoint = closestPoint(x_q2, q_point, "T")
    print("Second calculation closest point= " + str(xPoint + 1))
    plt.show()
    """
    return 0

prva()

#Question
## What do you notice about the relationship between the eigenvectors
# and the data? What happens to the eigenvectors if you change the data or add more
#points?

#Eigenvektorji bi se spremenenili,  sploh ce bi dodali mnogo točk na neko "premico".
# Ker bi bila najmanjša rekonstrukcijska napaka prisotna na drugačnem eigen vekotrju, bi imeli drugačne rezultate.

##Question 
# What happens to the reconstructed points? Where is the data projected to?
# Točke so projecirane na premico eigenvektorja, ki corresponda na največji eigenvalue.