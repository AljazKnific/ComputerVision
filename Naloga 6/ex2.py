from a6_utils import *
import numpy as np

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
    return odstej, C, U.T, S, VT 

def prva():
    
    tocke = np.loadtxt('data/points.txt')
    
    points = np.zeros((5,2))
    
    points[:,0] = tocke[:,0]
    points[:,1] = tocke[:,1]
    mu, C, U, S, VT = dualPCA(points.T)
    
    drawEllipse(mu, C)
    plt.scatter(points[:,0], points[:,1], marker='o')
    for i in range(points.shape[0]):
        plt.text(points[i,0], points[i,1], str(i + 1))
    
    for x in range(0, U.shape[0]):
        plt.plot([mu[0], mu[0] + U[x,0] * np.sqrt(S[x])], [mu[1], mu[1] + U[x,1] * np.sqrt(S[x])])
    
    y2 = np.dot(U,(points.T - mu))
    
    x_q2 = np.dot(U.T,y2) + mu
    
    plt.scatter(x_q2.T[:,0], x_q2.T[:,1], marker = '+')
    
    
    plt.show()
    
    return 0

prva()