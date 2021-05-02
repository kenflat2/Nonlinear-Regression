import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# String that defines which covar func to use
covarFuncStr = "sqrexp"
meanFuncStr = "zero"

X = np.array([])
Y = np.array([])
K = np.array([])
Kinv = np.array([])
originalX = np.array([])
originalY = np.array([])

Xflat = np.array([])
Yflat = np.array([])
Kflat = np.array([])
Kinvflat = np.array([])

dim = 0
embDim = 0
embInterval = 1

Xlen = 0
Ylen = 0

r = np.array([])

# number of optimization steps
optSteps = 20

numHP = {
    "exp" : 2,
    "sqrexp" : 2,
    "sqrexpf" : 3,
    "sqrexpo" : 3,
}

# HyperParameters
hp = np.ones(1 + numHP[covarFuncStr]) * 0.1 # array of hyperparameters for a given covar function

# Covar Func and derivative dictionaries - use hp array for each hyperparameter. Remember that hp[0] is reserved for prior variance, so start using hp[1] and up
covarDict = {
    "exp" : lambda x1, x2, t: hp[1] * np.exp( -hp[2] * (np.dot(x1-x2,x1-x2) ** (1/2))),                                 # hp[1] = tau, hp[2] = phi
    "sqrexp" : lambda x1, x2, t: hp[1] * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
    "sqrexpf" : lambda x1, x2, t: ((1 + hp[3]*t) ** (-1/2)) * hp[1] * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),        # hp[3] = delta(forgetting rate)
    
    # 1 : tau, 2 : phi_x, 3 : Vz*phi_a
    "sqrexpo" : lambda x1, x2, t: hp[1] * ((1 + 2*hp[3]*t) ** (-1/2)) * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),  
}
covarDerivDict = {
    "sqrexp" :  (   lambda x1, x2, t: np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
                    lambda x1, x2, t: -hp[1] * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2)
                ),
    "sqrexpf" : (   lambda x1, x2, t: ((1 + hp[3]*t) ** (-1/2)) * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
                    lambda x1, x2, t: ((1 + hp[3]*t) ** (-1/2)) * -hp[1] * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)) * np.dot(x1-x2,x1-x2),
                    lambda x1, x2, t: (hp[3]/2) * (((1 + hp[3]*t) ** (-3/2))) * hp[1] * np.exp( -hp[2] * np.dot(x1-x2,x1-x2))
                ),
    "sqrexpo" : (   lambda x1, x2, t: ((1 + 2*hp[3]*t) ** (-1/2)) * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
                    lambda x1, x2, t: -1 * np.dot(x1-x2,x1-x2) * hp[1] * ((1 + 2*hp[3]*t) ** (-1/2)) * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
                    lambda x1, x2, t: -1 * ((1 + 2*hp[3]*t) ** (-3/2)) * t * np.exp( -hp[2] * np.dot(x1-x2,x1-x2)),
                ),
}

meanDict = {
    "zero" : lambda x: np.zeros(dim)
}

availablePriorDict = {
    "none" : lambda x: 0,
    "half-normal" : lambda x: x * np.pi / np.sqrt(12),
    "ARD" : lambda x: -2 * np.exp(-(x**2)/np.pi)/np.pi
}

setPriorDict = {
    0 : "none"
}


# IMPORTANT NOTE: THIS CODE ONLY WORKS FOR SQREXP BASED COVAR, DON'T KNOW HOW IT GENERALIZES TO OTHERS YET
def covar(x1, x2, i1, i2, t):
    global r, covarDict, covarFuncStr, X, dim
    
    cov = covarDict[covarFuncStr](x1[:dim], x2[:dim], t)
    
    return cov * embCovarTerm(x1,x2)

def embCovarTerm(x1,x2):
    global r, numHP, covarFuncStr, X, dim
    # Adjust covariance to account for different timescales in time lags ( only works for sqexp at present, please be patient)
    embCov = 1
    for i in range(0,embDim):
        hpi = numHP[covarFuncStr]+1 + i
        embCov *= np.exp(-hp[hpi] * abs(x1[dim+i]-x2[dim+i]) / r[i])
    return embCov
    
# Returns partial derivative of covar over a given hyperparameter
def dCovardHp(i1, i2, h):
    global dim
    numDefaultHP = numHP[covarFuncStr]+1
    embHPi = h - numDefaultHP
    yi = dim + embHPi # index of embedding dimension
    
    global X, r
    if h < numDefaultHP:
        # print("embCovarTerm = ",embCovarTerm(X[i1],X[i2]))
        normalCovarTerm = covarDerivDict[covarFuncStr][h-1](X[i1,:dim],X[i2,:dim],abs(i1-i2))
        # print("Normal Covar Term = ", normalCovarTerm)
        return normalCovarTerm * embCovarTerm(X[i1],X[i2])
    elif h < len(hp):
        freshTerm = abs(X[i1,yi] - X[i2,yi]) / -r[embHPi]
        # print("freshTerm = ", freshTerm)
        return covar(X[i1], X[i2], i1, i2, abs(i1-i2)) * freshTerm
    print("RHU RHO RAGGY, not a valid hyperparameter! h = ", h)
    return 0

def setHP(ass):
    global hp
    if len(ass) == len(hp):
        hp = ass

def helpCovar():
    print("exp - exponential, sqrexp - squared exponential")

def setCovar(covstr):
    global setPriorDict
    if covstr in covarDict:
        global covarFuncStr, hp
        covarFuncStr = covstr
        hp = np.ones(1 + numHP[covarFuncStr]) * 0.5
        
        initPriors()
        print("Covariance function set to ", covstr)
    else:
        print("Unable to set covariance function, covar function remains ", covarFuncStr)

def initPriors():
    global setPriorDict, embDim
    
    for i in range(1,numHP[covarFuncStr]+1+embDim):
        if i not in setPriorDict:
            setPriorDict[i] = "none"
    print("Prior dict ", setPriorDict)

def setData(xd, yd):
    global X, Y, Xlen, Ylen, Xflat,Yflat, dim, r, embDim

    originalX = xd
    originalY = yd

    # all embedding nonsense is cleared when new data is applied
    embDim = 0
    hp = np.ones(1 + numHP[covarFuncStr]) * 0.1 # array of hyperparameters for a given covar function
    
    X = xd
    Y = yd
    Xlen = xd.shape[0]
    Ylen = yd.shape[0]

    dim = xd.shape[1]

    Xflat = X.flatten(order="F")
    Yflat = Y.flatten(order="F")

    createCovarMatrix()
    print("Data input success")

def calculateR():
    global X, r,dim

    r = np.amax(X[:,dim:], axis=0) - np.amin(X[:,dim:],axis=0)
    r[r==0] = 10**-10
    print("r ", r)
    """
    for x1 in X:
        for x2 in X:
            dist = la.norm(x1-x2)
            if dist > r:
                r = dist
    print("r = ", r, " versus approx ", 4.5 * np.sqrt(X.shape[1]))
    """
    # APPROXIMATE VERSION
    # Since data is normalized, the expected max distance should be around 4.5(2.25 st in any direction) times length of diagonal in unit hypercube    
    # r = 4.5 * np.sqrt(X.shape[1])    

def setPrior(varNum, str):
    global setPriorDict    
    setPriorDict[varNum] = str
    print("Prior dict ", setPriorDict)

def setTimeDelayEmbedding(assignment):
    global embDim, hp, X, embInterval, Xlen, Ylen, Y, originalX, originalY

    # delete previous embeddings if they exist
    if embDim != 0:
        X = originalX
        Y = originalY

    tmplen = X.shape[1]

    tmp = np.zeros([sum(x) for x in zip(X.shape,(0,sum(assignment)))])
    print("tmp ",tmp.shape)
    tmp[:,:X.shape[1]] = X
    X = tmp

    lag = 1
    newColInd = 0
    if len(assignment) != tmplen:
        print("Assigment list doesn't match the number of variables in data array! ",assignment)
        return
    else:
        # code that creates the lags
        for i in range(len(assignment)):
            for _ in range(assignment[i]):
                newCol = X[:-embInterval*lag,i]
                X[embInterval*lag:, tmplen + newColInd] = newCol
                newColInd += 1
                lag += 1
    X = X[embInterval*sum(assignment):]
    
    embDim = sum(assignment)
    hp = np.append(hp, np.ones(embDim) * 0.1)

    # update size of X and of Y
    Xlen = X.shape[0]
    Ylen = X.shape[0]
    Y = Y[-Xlen:]

    # call other methods to get set up
    calculateR()
    createCovarMatrix()
    initPriors()

    print("New X Dimensions ", X.shape)

def setTimeDelayInterval(i):
    global embInterval
    
    embInterval = i

def createCovarMatrix():
    global K, Kinv, Kflat, Kinvflat, Xlen
    K = np.zeros((Xlen,Xlen))
    for i in range(Xlen):
        for j in range(Xlen):
            K[i][j] = covar(X[i], X[j], i, j, abs(i-j))
    K = K + (hp[0] ** 2) * np.identity(Xlen) # add sigma to array, don't see any case where we need the default K
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
    global X

    ii = X.shape[0]
    
    C = np.zeros(Xlen)
    for i in range(Xlen):
        C[i] = covar(X[i],xin, i, ii,0)

    M = np.zeros(Y.shape)
    for i in range(Ylen):
        M[i] = meanDict[meanFuncStr](X[i])

    # print(C.shape," ", CM.shape," ", Y.shape)
    pred = meanDict[meanFuncStr](xin) + C @ Kinv @ (Y - M)
    predVar = covar(xin,xin,ii,ii, 0) - C @ Kinv @ np.transpose(C)

    return (pred, predVar)

def hyperParamOptimize(steps=20,yind=0):
    global hp, optSteps
    
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
    
    maxCount = steps
    count = 0

    rhoplus = 1.2 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1

    hypermin = 1 * (10**-8)
    hypermax = 10

    i = 0

    x = np.zeros((maxCount,1))
    y = np.zeros((maxCount,1))

    grads = np.zeros((2,maxCount))
    # print(grads.shape)

    deltaPrev = np.ones(len(hp)) * 0.1 # low initial delta value, this modifies vars directly
    gradPrev = deltaPrev
    while la.norm(gradPrev) > 0.001  and count < maxCount:
        grad = hyperParamGradient()
        grad = grad / la.norm(grad)# np.abs(grad) # NORMALIZE, because rprop ignores magnitude
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

        # floor and ceiling on the hyperparameters
        hp = np.clip(hp+dweights, hypermin, hypermax)
        hp[1+numHP[covarFuncStr]:] = np.clip(hp[1+numHP[covarFuncStr]:], hypermin, 1) # ceil for emb lengthscales 

        if covarFuncStr in list("sqrexpf"):
            hp[3] = min(hp[3],1)

        x[i] = hp[0]
        y[i] = hp[1]
        grads[0][i] = grad[0]
        grads[1][i] = grad[1]
        i += 1

        createCovarMatrix()

        print("Hp: ", hp, " # ", count)
    fig3, ax3 = plt.subplots()
    ax3.quiver(x,y,grads[0],grads[1])#grads[:,0],grad[:,1])
    ax3.set_xlabel("Sigma")
    ax3.set_ylabel("Tau")
    plt.show()

    print("=SPLIT(\"",hp[3],",",hp[4],",",hp[5],",",hp[6],"\",\",\")")
    
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

def hyperParamLikelihood(yind=0):
    # print("Yflat ",Yflat.shape, " Kinvflat ", Kinvflat.shape)
    v = np.pi / np.sqrt(12)
    
    # print(Y.shape, K.shape, Kinv.shape, yind)
          
    # print(la.slogdet(K))
    # print(Y[:,yind].T @ Kinv @ Y[:,yind])
    priorPenalty = np.log(2) - hp[2]**2 / (2*v) + np.log(np.sqrt(2*np.pi*v))
    return -0.5 * (la.slogdet(K)[1] + Y[:,yind].T @ Kinv @ Y[:,yind]) + priorPenalty

    # return -0.5 *( np.transpose(Y[:,yind]) @ Kinv @ Y[:,yind] + np.log(la.norm(K)) + X.shape[0] * np.log(2*np.pi)) - np.log(2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v))
    
    # print(np.log(la.norm(K)), np.transpose(Y) @ Kinv @ Y) 
    # print(np.transpose(Yflat) @ Kinvflat @ Yflat,np.log(la.norm(Kflat)),Xflat.shape[0] * np.log(2*np.pi),- 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v))
    # return -0.5 *( np.transpose(Y) @ Kinv @ Y + np.log(la.norm(K)) + X.shape[0] * np.log(2*np.pi)) - 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v)
    # return -0.5 *( np.transpose(Yflat) @ Kinvflat @ Yflat + np.log(la.norm(Kflat)) + Xflat.shape[0] * np.log(2*np.pi)) - 2 * np.exp(-phi**2/(2*v))/np.sqrt(2*np.pi*v)
    # return -0.5 * (np.log(la.norm(Kflat)) + np.transpose(Yflat) @ Kinvflat @ Yflat )
    # return -0.5 * (np.log(la.norm(K)) + np.transpose(Y) @ Kinv @ Y )

def hyperParamGradient(yind=0):
    global hp
    # calculate gradient of K for hyperparams, requires custom partial derivative calculations
    grad = np.zeros(len(hp))
    
    # first find gradient for variance, since that is independent of the covariance function
    dKdSig = np.identity(Xlen) * hp[0]
    # print(dKdSig.shape, Y[:,yind].T.shape, Kinv.shape, K.shape)
    dSigma = 0.5 * ( Y[:,yind].T @ Kinv @ dKdSig @ Kinv @ Y[:,yind] - np.trace(Kinv @ dKdSig))
    grad[0] = dSigma

    # Then find gradient for all other values
    dKdHP = np.zeros((X.shape[0],X.shape[0]))
    for h in range(1,len(hp)):
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                dKdHP[i][j] = dCovardHp(i,j,h) # covarDerivDict[covarFuncStr][h](X[i],X[j],abs(i-j))

        #                                                                                       \/ get appropriate prior function and pass in current hp(we love ugly code)
        dHP = 0.5 * ( Y[:,yind].T @ Kinv @ dKdHP @ Kinv @ Y[:,yind] - np.trace(Kinv @ dKdHP)) - availablePriorDict[setPriorDict[h]](hp[h])

        grad[h] = dHP

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
    
