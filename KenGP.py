import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Global Variables
tau = 0.5
phi = 0.5
sigma = 0.001
delta = 0.06

covarFuncStr = "sqrexp"
meanFuncStr = "zero"

X = np.array([])
Y = np.array([])
K = np.array([])
Kinv = np.array([])

Xflat = np.array([])
Yflat = np.array([])
Kflat = np.array([])
Kinvflat = np.array([])

dim = 0

Xlen = 0
Ylen = 0

covarDict = {
    "exp" : lambda x1,x2: tau * np.exp( -phi * (np.dot(x1-x2,x1-x2) ** (1/2))),
    "sqrexp" : lambda x1,x2: tau * np.exp( -phi * np.dot(x1-x2,x1-x2)),
    "sqrexpf" : lambda x1, x2, t: ((1 + delta*t) ** (-1/2)) * tau * np.exp( -phi * np.dot(x1-x2,x1-x2)),
    "sqrexpo" : lambda x1, x2, t: ((1 + 2*phi*sigma*t) ** (-1/2)) * tau * np.exp( -phi * np.dot(x1-x2,x1-x2)),
}
covarDerivDict = {
    "sqrexp" : ( lambda x1, x2: np.exp( -phi * np.dot(x1-x2,x1-x2)),
                 lambda x1, x2: -tau * np.exp( -phi * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2)),
    "sqrexpf" : ( lambda x1, x2: np.exp( -phi * np.dot(x1-x2,x1-x2)),
                 lambda x1, x2: -tau * np.exp( -phi * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2)),
    "sqrexpo" : ( lambda x1, x2: np.exp( -phi * np.dot(x1-x2,x1-x2)),
                 lambda x1, x2: -tau * np.exp( -phi * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2)),
}
meanDict = {
    "zero" : lambda x: np.zeros(dim)
}
numHyperParam = {
    "exp" : 2,
    "sqrexp" : 2,
    "sqrexpf" : 3,
    "sqrexpo" : 2,
}
        
def helpCovar():
    print("exp - exponential, sqrexp - squared exponential")

def setCovar(covstr):
    if covstr in covarDict:
        global covarFuncStr
        covarFuncStr = covstr
        print("Covariance function set to ", covstr)
    else:
        print("Unable to set covariance function")

def setData(xd, yd):
    global X, Y, Xlen, Ylen, Xflat,Yflat, dim
    
    X = xd
    Y = yd
    Xlen = xd.shape[0]
    Ylen = yd.shape[0]

    dim = xd.shape[1]

    Xflat = X.flatten(order="F")
    Yflat = Y.flatten(order="F")
    
    createCovarMatrix()
    print("Data input success")

def createCovarMatrix():
    global K, Kinv, Kflat, Kinvflat
    K = np.zeros((Xlen,Xlen))
    for i in range(Xlen):
        for j in range(Xlen):
            if covarFuncStr in ("sqrexpf","sqrexpo"):
                K[i][j] = covarDict[covarFuncStr](X[i],X[j],abs(i-j))
            else:
                K[i][j] = covarDict[covarFuncStr](X[i],X[j])
    K = K + (sigma ** 2) * np.identity(Xlen) # add sigma to array, don't see any case where we need the default K
    # print("K = ", K, " tau ", tau, " phi ", phi)
    Kinv = la.inv(K) # invert K immediately, get it outta the way
    # make flattened versions
    """
    fig7 = plt.figure(7)
    plt.imshow(K,cmap="hot")
    fig8 = plt.figure(8)
    plt.imshow(Kinv,cmap="hot")
    
    Kflat = np.zeros((Xflat.shape[0],Xflat.shape[0]))
    for i in range(Xflat.shape[0]):
        for j in range(Xflat.shape[0]):
            Kflat[i][j] = covarDict[covarFuncStr](Xflat[i],Xflat[j])
    # print("Kflat = ", Kflat, " tau ", tau, " phi ", phi)
    Kflat = Kflat + (sigma ** 2) * np.identity(Xflat.shape[0])
    Kinvflat = la.inv(Kflat)
    
    fig9 = plt.figure(9)
    plt.imshow(Kflat,cmap="hot")
    fig10 = plt.figure(10)
    plt.imshow(Kinvflat,cmap="hot")
    """
    plt.show()

def predict(xin):
    #print(X)
    C = np.zeros(Xlen)
    for i in range(Xlen):
        if covarFuncStr in ("sqrexpf","sqrexpo"):
            C[i] = covarDict[covarFuncStr](xin, X[i],0)
        else:
            C[i] = covarDict[covarFuncStr](xin, X[i])

    M = np.zeros(Y.shape)
    for i in range(Ylen):
        M[i] = meanDict[meanFuncStr](X[i])

    # print(C.shape," ", CM.shape," ", Y.shape)
    pred = meanDict[meanFuncStr](xin) + C @ Kinv @ (Y - M)
    if covarFuncStr in ("sqrexpf","sqrexpo"):
        predVar = covarDict[covarFuncStr](xin,xin,0) - C @ Kinv @ np.transpose(C)
    else:
        predVar = covarDict[covarFuncStr](xin,xin) - C @ Kinv @ np.transpose(C)
    
    return (pred, predVar)

def hyperParamOptimize():
    global tau, phi, sigma
    
    # time for RPROP
    """
    tauRange = np.arange(0.05,1.0,.1)
    phiRange = np.arange(0.05,1.0,.1)
    likelihoods = np.zeros((tauRange.shape[0],phiRange.shape[0]))
    gradients = np.zeros((tauRange.shape[0],phiRange.shape[0],2))

    for t in range(tauRange.shape[0]):
        for p in range(phiRange.shape[0]):
            
            tau = tauRange[t]
            phi = phiRange[p]
            print(tau," ",phi)
            createCovarMatrix()
            likelihood = hyperParamLikelihood()
            gradient = hyperParamGradient()
            print("Grad = ",gradient, " prob = ", likelihood)
            # print("Likelihood = ",likelihood)
            # gradVals[t][p][0] = tauRange[t]
            # gradVals[t][p][1] = phiRange[p]
            likelihoods[t][p] = likelihood
            gradients[t][p] = gradient / la.norm(gradient)

    fig2, ax2 = plt.subplots()
    x,y = np.meshgrid(tauRange, phiRange)
    print(x.shape,y.shape)
    ax2.contour(x,y, likelihoods,levels=50)
    ax2.set_xlabel("Tau")
    ax2.set_ylabel("Phi")

    fig3, ax3 = plt.subplots()
    ax3.quiver(x,y,gradients[:,:,0],gradients[:,:,1])
    ax2.set_xlabel("Tau")
    ax2.set_ylabel("Phi")
    
    # ax2 = fig2.gca(projection = "3d")
    # ax2.scatter(gradVals[:,:,0],gradVals[:,:,1],gradVals[:,:,2])
    plt.show()
    """
    
    maxCount = 20
    count = 0

    rhoplus = 1.2 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1

    hypermin = 1 * (10**-6)
    hypermax = 10

    i = 0

    x = np.zeros((maxCount,1))
    y = np.zeros((maxCount,1))

    grads = np.zeros((2,maxCount))
    print(grads.shape)

    deltaPrev = np.ones((1+numHyperParam[covarFuncStr])) * 0.5 # low initial delta value, this modifies vars directly
    gradPrev = deltaPrev
    while la.norm(gradPrev) > 0.00001  and count < maxCount:
        grad = hyperParamGradient()
        grad = grad / la.norm(grad) # NORMALIZE, because rprop ignores magnitude
        print("Gradient: ",grad)
        print("Likelihood: ", hyperParamLikelihood())

        s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
        spos = np.ceil(s) # 0 for - vals, 1 for + vals
        sneg = -1 * (spos - 1) # oh dear i am such a clever fellow

        delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
        dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

        deltaPrev = delta
        gradPrev = grad
        count += 1

        sigma = min(max(sigma+dweights[0],hypermin),hypermax)
        tau = min(max(tau+dweights[1],hypermin),hypermax)
        phi = min(max(phi+dweights[2],hypermin),hypermax)

        x[i] = sigma
        y[i] = tau
        grads[0][i] = grad[0]
        grads[1][i] = grad[1]
        i += 1

        createCovarMatrix()

        print("Tau ", tau, " Phi ", phi, " Sigma ", sigma, " count ", count)
    fig3, ax3 = plt.subplots()
    ax3.quiver(x,y,grads[0],grads[1])#grads[:,0],grad[:,1])
    ax3.set_xlabel("Sigma")
    ax3.set_ylabel("Tau")
    plt.show()
    """
    hyper = np.zeros((10,10,10,4),dtype=np.float)
    for t in range(0,10):
        for p in range(0,10):
            for s in range(0,10):
                tau = t + 0.5
                phi = p + 0.5
                sigma = s + 0.5
                error = np.log(la.norm(Y[4] - predict(X[3])[0]))
                # print("Error ",tau," ",phi," ",sigma," ", error)
                hyper[t][p][s][0] = tau
                hyper[t][p][s][1] = phi
                hyper[t][p][s][2] = sigma
                hyper[t][p][s][3] = -1 * error

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(hyper[:,:,:,0],hyper[:,:,:,1],hyper[:,:,:,2],c=hyper[:,:,:,3],cmap="hot")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Phi")
    ax.set_zlabel("Sigma")
    plt.show()
    """

def hyperParamLikelihood():
    # print("Yflat ",Yflat.shape, " Kinvflat ", Kinvflat.shape)
    yind = 1
    v = np.pi / np.sqrt(12)
    
    print(la.slogdet(K), Y[:,yind].T @ Kinv @ Y[:,yind])
    priorPenalty = np.log(2) - phi**2 / (2*v) + np.log(np.sqrt(2*np.pi*v))
    return -0.5 * (la.slogdet(K)[1] + Y[:,yind].T @ Kinv @ Y[:,yind]) + priorPenalty

    # return -0.5 *( np.transpose(Y[:,yind]) @ Kinv @ Y[:,yind] + np.log(la.norm(K)) + X.shape[0] * np.log(2*np.pi)) - np.log(2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v))
    
    # print(np.log(la.norm(K)), np.transpose(Y) @ Kinv @ Y) 
    # print(np.transpose(Yflat) @ Kinvflat @ Yflat,np.log(la.norm(Kflat)),Xflat.shape[0] * np.log(2*np.pi),- 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v))
    # return -0.5 *( np.transpose(Y) @ Kinv @ Y + np.log(la.norm(K)) + X.shape[0] * np.log(2*np.pi)) - 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v)
    # return -0.5 *( np.transpose(Yflat) @ Kinvflat @ Yflat + np.log(la.norm(Kflat)) + Xflat.shape[0] * np.log(2*np.pi)) - 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v)
    # return -0.5 * (np.log(la.norm(Kflat)) + np.transpose(Yflat) @ Kinvflat @ Yflat )
    # return -0.5 * (np.log(la.norm(K)) + np.transpose(Y) @ Kinv @ Y )

def hyperParamGradient():
    yind = 1
    # calculate gradient of K for hyperparams, requires custom partial derivative calculations
    grad = np.zeros(1 + numHyperParam[covarFuncStr])
    
    # first find gradient for variance, since that is independent of the covariance function
    dKdSig = np.identity(Xlen) * sigma
    dSigma = 0.5 * ( Y[:,yind].T @ Kinv @ dKdSig @ Kinv @ Y[:,yind] - np.trace(Kinv @ dKdSig))
    grad[0] = dSigma

    # Then find gradient for all other values
    dKdTau = np.zeros((X.shape[0],X.shape[0]))
    dKdPhi = np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dKdTau[i][j] = covarDerivDict[covarFuncStr][0](X[i],X[j])
            dKdPhi[i][j] = covarDerivDict[covarFuncStr][1](X[i],X[j])

    dTau = 0.5 * ( Y[:,yind].T @ Kinv @ dKdTau @ Kinv @ Y[:,yind] - np.trace(Kinv @ dKdTau)) - tau * np.pi / np.sqrt(12)
    dPhi = 0.5 * ( Y[:,yind].T @ Kinv @ dKdPhi @ Kinv @ Y[:,yind] - np.trace(Kinv @ dKdPhi)) - phi * np.pi / np.sqrt(12)

    grad[1] = dTau
    grad[2] = dPhi

    return grad
    
    """
    yind = 1
    # determine the gradient of hyperparams to do gradient descent, only works on 2 vars for now
    aa = (Kinv @ Y[:,yind]) @ np.transpose( Kinv @ Y[:,yind] )

    # calculate gradient of K for hyperparams, requires custom partial derivative calculations
    dKdTau = np.zeros((Xlen,Ylen))
    dKdPhi = np.zeros((Xlen,Ylen))
    for i in range(Xlen):
        for j in range(Ylen):
            dKdTau[i][j] = covarDerivDict[covarFuncStr][0](X[i],X[j])
            dKdPhi[i][j] = covarDerivDict[covarFuncStr][1](X[i],X[j])

    # bring it all together
    dtau = 0.5 * np.trace((aa - Kinv) @ dKdTau )
    dphi = 0.5 * np.trace((aa - Kinv) @ dKdPhi )

    return np.array( [ dtau, dphi ] )
    """
    
