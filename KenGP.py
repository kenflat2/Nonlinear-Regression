import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Global Variables
tau = 5
phi = 5
sigma = 1

covarFuncStr = "sqrexp"
meanFuncStr = "zero"

X = np.array([])
Y = np.array([])
K = np.array([])
Kinv = np.array([])

Xlen = 0
Ylen = 0

covarDict = {
    "exp" : lambda x1,x2: tau * np.exp( -phi * (np.dot(x1-x2,x1-x2) ** (1/2))),
    "sqrexp" : lambda x1,x2: tau * np.exp( -phi * np.dot(x1-x2,x1-x2)),
}
covarDerivDict = {
    "sqrexp" : ( lambda x1, x2: np.exp( -phi * np.dot(x1-x2,x1-x2)),
                 lambda x1, x2: -tau * np.exp( -phi * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2)),
}

meanDict = {
    "zero" : lambda x: np.zeros(X.shape[1])
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
    global X, Y, Xlen, Ylen
    
    X = xd
    Y = yd
    Xlen = xd.shape[0]
    Ylen = yd.shape[0]
    print("Xlen ",Xlen, " Ylen ", Ylen)
    createCovarMatrix()
    print("Data input success")

def createCovarMatrix():
    global K, Kinv
    K = np.zeros((Xlen,Ylen))
    for i in range(Xlen):
        for j in range(Ylen):
            cov = covarDict[covarFuncStr](X[i],Y[j])
            K[i][j] = covarDict[covarFuncStr](X[i],Y[j])
    Kinv = la.inv(K) # invert K immediately, get it outta the way

def predict(xin):
    #print(X)
    C = np.zeros(Xlen)
    for i in range(Xlen):
        C[i] = covarDict[covarFuncStr](xin, X[i])

    M = np.zeros(Y.shape)
    for i in range(Ylen):
        M[i] = meanDict[meanFuncStr](X[i])

    CM = la.inv(K + (sigma ** 2) * np.identity(Xlen))
    # print(C.shape," ", CM.shape," ", Y.shape)
    pred = meanDict[meanFuncStr](xin) + C @ CM @ (Y - M)
    predVar = covarDict[covarFuncStr](xin,xin) - C @ CM @ np.transpose(C)
    
    return (pred, predVar)

def hyperParamOptimize():
    global tau, phi, sigma
    
    # time for RPROP
    tauRange = np.arange(0.5,20,2)
    phiRange = np.arange(0.5,20,2)
    gradVals = np.zeros((tauRange.shape[0],phiRange.shape[0],2))
    for t in range(tauRange.shape[0]):
        for p in range(phiRange.shape[0]):
            tau = tauRange[t]
            phi = phiRange[p]
            createCovarMatrix()
            gradVals[t][p] = hyperParamLikelihood()

    plt.figure()
    plt.quiver(tauRange,phiRange,gradVals[:,:,0],gradVals[:,:,1])
    plt.show()
    """
    maxCount = 40
    count = 0

    rhoplus = 1.2 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1

    deltaPrev = np.ones((2)) * 0.5 # low initial delta value, this modifies vars directly
    gradPrev = deltaPrev
    while la.norm(gradPrev) > 0.00001  and count < maxCount:
        grad = -1 * hyperParamGradient()
        grad = grad / la.norm(grad) # NORMALIZE, because rprop ignores magnitude
        print(grad)

        s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
        spos = np.ceil(s) # 0 for - vals, 1 for + vals
        sneg = -1 * (spos - 1) # oh dear i am such a clever fellow

        delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
        dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

        deltaPrev = delta
        gradPrev = grad
        count += 1

        tau += dweights[0]
        phi += dweights[1]
        createCovarMatrix()

        print("Tau ", tau, " Phi ", phi, " count ", count)
        """
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
    return -0.5 *( np.transpose(Y) @ Kinv @ Y + np.log(la.norm(K)) + Xlen * np.log(2*np.pi))

def hyperParamGradient():
    # determine the gradient of hyperparams to do gradient descent, only works on 2 vars for now
    aa = Kinv @ Y @ np.transpose( Kinv @ Y )

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
