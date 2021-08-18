import math
import numpy as np
import numpy.linalg as la
import numpy.random as rand
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from sklearn.manifold import MDS
from numpy import genfromtxt

def delayEmbed(Xin, Yin,assignment,embInterval):
    
    tmplen = Xin.shape[1]

    tmp = np.zeros([sum(x) for x in zip(Xin.shape,(0,sum(assignment)))])
    tmp[:,:Xin.shape[1]] = Xin
    Xin = tmp

    lag = 1
    newColInd = 0
    if len(assignment) != tmplen:
        print("Assigment list doesn't match the number of variables in data array! ",assignment)
        return
    else:
        # code that creates the lags
        for i in range(len(assignment)):
            for _ in range(assignment[i]):
                newCol = Xin[:-embInterval*lag,i]
                Xin[embInterval*lag:, tmplen + newColInd] = newCol
                newColInd += 1
                lag += 1
    Xin = Xin[embInterval*sum(assignment):]
    Yin = Yin[embInterval*sum(assignment):]
    
    # Yin = Yin[-X.shape[0]:]
    
    return (Xin, Yin)

def hprime(x):
    n = x.shape[0]
    d = x.shape[1]
    
    M = int(d+d*(d+1)/2) + d # number of monomials and binomials
    
    hx = np.zeros((n,M))
    hx[:,0:d] = x
    ind = d
    for i in range(d):
        xi = x[:,i]
        for j in range(i,d):
            xj = x[:,j]
            hx[:,ind] = np.multiply(xi, xj)
            ind += 1
            
    hx[:,M-d:] = np.exp(x)
    
    print(hx)
            
    return hx

def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

end = 2**8
tlen = 2 ** 9
print("Stepsize = {st}".format(st=end/tlen))
trainToTest = 0.5 # between 0 and 1
t = np.linspace(0, end, num=tlen)

# MAKE SURE TO UPDATE THE DIMENSION WHEN SWITCHING ATTRACTORS
dim = 1
ndrivers = 1
# t0 = np.array([0.5])
t0 = np.ones(dim) * 0.3333 # np.array([0,5,15]) * 1 # np.zeros(dim)
t0[0] += 0.1

# Logistic Map

r = lambda t : 3.35 + 0.6 * np.sin(0.7*2*np.pi*t/tlen)
# r = lambda t : 0.5 * t / tlen + 3.5
states = np.zeros((tlen,1))
states[0,0] = t0
for t in range(1,tlen):
    states[t,0] = r(t) * states[t-1,0] * (1 - states[t-1,0])
Xr = states

""" UPDATE DRIVERS HERE """
digiDrivers = [r]

gtsr = np.zeros((Xr.shape[0], ndrivers))
for ind in range(len(digiDrivers)):
    tmp = np.fromfunction(lambda i : digiDrivers[ind](i), (Xr.shape[0],) , dtype = float)# time series of gmax
    gtsr[:,ind] = tmp

# Delay Embed
X, _ = delayEmbed(Xr, Xr, [2]*dim,1)
# X, _ = delayEmbedUnitary(Xr, Xr, 3,1) # << Seems to always suck, even though it should be better...

print(Xr.shape)
gtsr = standardize(gtsr[:X.shape[0]])
print(X.shape,gtsr.shape)

np.set_printoptions(precision=4, suppress=True)
Xst = standardize(X)

zprime = Xst
# zprime = hprime(Xst)
print(zprime.shape)

# replace this with SVD?
c = np.cov(zprime.T, bias=False, rowvar=True)
# print("Covariance = ", c, c.shape)

eigval, eigvec = la.eigh(c)
# print("Eigenstuff = ", eigval, eigvec)
print("Eigenvals(shouldn't be near 0) ", eigval)

diagEigVal = np.diag((eigval+1e-7) ** -0.5)

print(zprime.shape, diagEigVal.shape, eigvec.T.shape)
z = zprime @ (eigvec @ diagEigVal)
# print("z = ", z)
print("Mean test(should be 0)", np.mean(z))
print("Covariance test(should be I): \n",np.cov(z.T))

zdot = z[1:,:] - z[:-1,:]

"""
# print((zdot @ zdot.T).round(4))
covzdot = np.cov(zdot.T)
# print(covzdot.shape)
eigValDot, eigVecDot = la.eigh(covzdot)
print("EigenVectors = ", eigVecDot)

a = eigVecDot[:,np.argsort(eigValDot)[0]] # eigVecDot.sort(key=eigValDot)[0]
# print(a)
yt = a @ z.T
print(a.shape, z.shape)
"""

embedding = MDS(n_components = 3,metric=False)
X_transformed = embedding.fit_transform(zdot)

figNLDR = plt.figure(1,figsize=(10,10))
axNLDR = figNLDR.gca(projection="3d")
axNLDR.scatter(X_transformed[:,0],X_transformed[:,1],X_transformed[:,2])
# axNLDR.scatter(X_transformed[:,3],X_transformed[:,4],X_transformed[:,5])
plt.show()
