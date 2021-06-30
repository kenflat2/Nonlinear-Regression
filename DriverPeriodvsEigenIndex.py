import numpy as np
import numpy.linalg as la
import numpy.random as rand
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint


# Slow Feature Analysis

def LorenzP(xi,t, rho, sigma, beta):
    
    (x,y,z) = xi
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def RosslerP(xi, t, a, b, c):    
    (x,y,z) = xi

    dx = -y - z
    dy = x + a * y
    dz = b + z * ( x - c )

    return np.array( [dx,dy,dz] )

def hprime(x):
    n = x.shape[0]
    d = x.shape[1]
    
    M = int(d+d*(d+1)/2) # number of monomials and binomials
    
    hx = np.zeros((n,M))
    hx[:,0:d] = x
    ind = d
    for i in range(d):
        xi = x[:,i]
        for j in range(i,d):
            xj = x[:,j]
            hx[:,ind] = np.multiply(xi, xj)
            ind += 1
            
    return hx

def standardize(x):
    return (x - np.mean(x, axis=0)) # / np.std(x, axis=0)

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

end = 500
tlen = 2**10
print("Stepsize = {st}".format(st=end/tlen))
trainToTest = 0.5 # between 0 and 1
t = np.linspace(0, end, num=tlen)

# MAKE SURE TO UPDATE THE DIMENSION WHEN SWITCHING ATTRACTORS
dim = 3
# t0 = np.array([0.5])
t0 = np.array([0,5,15]) * 1 # np.ones(dim) * 0.3333  # np.zeros(dim)
t0[0] += 0.1

for b in range(-30, 30):
    per = (2**b)

    """ LORENZ
    rho = lambda t : 28 + 4 * np.sin( per * 2*np.pi * t / (tlen-2))# (2*np.heaviside(t-500, 1)-np.heaviside(t-1000, 1)) # rho = 28.0
    # sigma = 10       # sigma = 10.0
    sigma = lambda t : 10.0 # np.sin( 4 * 2*np.pi * t / (tlen-2))
    beta = lambda t : 8.0 / 3.0  # beta = 8.0 / 3.0
    
    largs = lambda t : (rho(t), sigma(t), beta(t))
    
    states = np.zeros((tlen,3))
    states[0] = t0
    for i in range(1, tlen ):
        # print(largs(i))
        states[i] = odeint(LorenzP,states[i-1],np.array([t[i-1],t[i]]),args=largs(i))[1,:]
    X = states
    """

    # Rossler
    ap = lambda t : 0.2 + 0.1 * np.sin( per * 2*np.pi * t / (tlen-2)) # (2*np.heaviside(t-500, 1)-np.heaviside(t-1000, 1)) # rho = 28.0
    # sigma = 10       # sigma = 10.0
    bp = lambda t : 0.2 # np.sin( 4 * 2*np.pi * t / (tlen-2))
    cp = lambda t : 5.7 # beta = 8.0 / 3.0

    largs = lambda t : (ap(t), bp(t), cp(t))

    states = np.zeros((tlen,3))
    states[0] = t0
    for i in range(1, tlen ):
    # print(largs(i))
        states[i] = odeint(RosslerP,states[i-1],t[i-1:i+1],args=largs(i))[1,:]
    Xr = states

    X, _ = delayEmbed(Xr, Xr, [3,3,3],1)

    np.set_printoptions(precision=4, suppress=True)
    # print("X = ", X.shape)
    Xst = standardize(X)
    # print(Xst.shape, hp.shape)
    # zprime = Xst
    zprime = hprime(Xst)
    c = np.cov(zprime.T, bias=False)

    eigval, eigvec = la.eigh(c)

    diagEigVal = np.diag((eigval+1e-10) ** -0.5)

    z = zprime @ (eigvec @ diagEigVal)

    zdot = z[1:,:] - z[:-1,:]

    # print((zdot @ zdot.T).round(4))
    covzdot = np.cov(zdot.T)
    # print(covzdot.shape)
    eigValDot, eigVecDot = la.eigh(covzdot)

    a = eigVecDot[:,np.argsort(eigValDot)[0]] # eigVecDot.sort(key=eigValDot)[0]
    yt = a @ z.T

    # Idea - write function that check similarity between true and SFA'd time series

    gts = np.fromfunction(lambda i : ap(i), yt.shape , dtype = float)# time series of gmax
    cutoff = 5000

    nVec = z.shape[1]
    gtsStnd = (gts - np.mean(gts)) / np.std(gts)

    diffs = np.zeros(nVec)

    for e in range(nVec):
        ae = eigVecDot[:,np.argsort(eigValDot)[e]] @ z.T
        aeStnd = ae - np.mean(ae)
        aeStnd = aeStnd / la.norm(aeStnd)
        delta = gtsStnd @ aeStnd.T
        aeScld = aeStnd * delta
        
        diffs[e] = la.norm(gtsStnd - aeScld)
        # print(delta)

    K = 3
    diffSrtd = np.argsort(diffs)
    print("Best Index for Period of {p}(2^{bb}) is {ind}".format(p=per,ind=diffSrtd[0],bb=b))
