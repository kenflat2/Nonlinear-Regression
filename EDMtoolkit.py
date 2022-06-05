import numpy as np
import math
import numpy.linalg as la
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

epsilon = 2e-10

def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def nearestNeighbors(s0, S, n):
    orderedNeighbors = np.argsort(la.norm(s0 - S[:-1],axis=1))    
    return orderedNeighbors[1:n+1]

def removeNANs(TS, t=None):
    notNA = np.all(~np.isnan(TS),axis=1)

    D = TS[notNA]
    if t is not None:
        t = t[notNA]
        return (D, t)

    return D

def distanceMatrix(X):
    n = X.shape[0]
    
    distance_matrix = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            distance_matrix[i,j] = la.norm(X[i]-X[j])
            
    return distance_matrix

# create a delay embeddding vector from a given UNIVARIATE time series.
def delayEmbed(D, predHorizon, nLags, embInterval, t = None, removeNAs=True):
    
    totalRows = D.shape[0] + predHorizon + embInterval * nLags
    A = np.zeros((totalRows, 2 + nLags))
    
    A[:D.shape[0],0] = D.flatten()
    
    for i in range(1, 2 + nLags):
        lower = predHorizon + (i - 1) * embInterval
        upper = lower + D.shape[0]
        A[lower:upper, i] = D.flatten()
    
    rowsLost = predHorizon + nLags * embInterval
    if rowsLost != 0:
        B = A[rowsLost : -rowsLost]
        if t is not None:
            t = t[ : -rowsLost]
    else: 
        B = A
    
    if removeNAs:
        notNA = np.all(~np.isnan(B),axis=1)

        B = B[notNA]
        if t is not None:
            print(t.shape, notNA.shape)
            t = t[notNA]
    
    if t is None:
        return (B[:,1:], B[:,0, None])
    else:
        return (B[:,1:], B[:,0, None], t)

### MULTIVARIATE DELAY EMBEDDING ###
def delayEmbedM(Xin, Yin,assignment,embInterval):
    
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

# Lyapunov Edition
# calculate dominant finite time lyapunov exponent of a system
def lyapunovExp(S):
    Lexp = 0
    n = 0 # S.shape[0]-1
    for i in range(S.shape[0]-1):

        # make sure this element is not null
        if arrayContainsNull(S[i]):
            continue

        # get neighbors
        nearNeighborsIndices = nearestNeighbors(S[i], S, 1)
        for nni in nearNeighborsIndices:
            # Make sure the next elements aren't null either
            if arrayContainsNull(S[i+1]) or arrayContainsNull(S[nni+1]):
                continue
            n += 1
            
            d = la.norm(S[i] - S[nni])
            dp = la.norm(S[i+1] - S[nni+1])
            
            d = max(d, epsilon)
            dp = max(dp, epsilon)
            
            fprime = dp / d
            if math.isnan(fprime):
                print("Sowind again: ", S[i], S[nni], d, dp)
                pass
            Lexp += np.log(fprime) # / la.norm(S[i] - S[nni])
    return Lexp / n # geometric mean - seems like lyapunov right?


def arrayContainsNull(A):
    return ~np.all(~np.isnan(A))

# False Nearest Neighbors Plot
def FNNplot(Xr, l=10, st=3):
    dim = Xr.shape[1]
    # figFNN, axFNN = plt.subplots(2 * c,figsize=(16, 3*(2*c)))
    figFNN, axFNN = plt.subplots(dim, figsize=(6, 3*dim))
    # figFNN = plt.figure(figsize=(12, 8))
    # axFNN = figFNN.add_subplot()

    for d in range(dim):
        lyapExps = np.zeros(l+1)
        for s in range(1, st+1, 1):
            for i in range(1,l+2):
                Y, _ = delayEmbed(Xr[:,d,None], 0, i, s) # individual axis version
                # Y, _ = delayEmbed(Xr[::c], Xr[::c], [i]*dim,s)
                # Y, _ = delayEmbedUnitary(Xr[::c], Xr[::c], i,s)
                lyapExps[i-1] = lyapunovExp(Y)
            # print(lyapExps)
            if dim == 1:
                axFNN.plot(range(2,l+3), lyapExps, label="{e}".format(e=s))
            else:
                axFNN[d].plot(range(2,l+3), lyapExps, label="{e}".format(e=s))
        
        if dim == 1:
            axFNN.legend()
            axFNN.set_xlabel("Embedding Dimension")
            # axFNN[c-1].set_title("Slice = {ind}".format(ind=c))
            axFNN.set_ylabel("Lyapunov Exponent")
        else:
            axFNN[d].legend()
            axFNN[d].set_xlabel("Embedding Dimension")
            # axFNN[c-1].set_title("Slice = {ind}".format(ind=c))
            axFNN[d].set_ylabel("Lyapunov Exponent")

    plt.show()

def nearestNeighborsPrediction(state):
    neighborIndexes = nearestNeighbors(state, numNeighbors)
    pred1neigh = list(map(lambda i: trainStates[i+1], neighborIndexes))
    return sum(pred1neigh) / numNeighbors

def timescaleInfo(X, Y, x, theta):
    norms = la.norm(X-x,axis=1)
    d = np.mean(norms) # d = np.mean(norms) # 
    
    W = np.exp(-1 * theta * norms / d)
    # print(X.shape, np.diag(W).shape, Y.shape)
    H = la.inv(X.T @ np.diag(W) @ X) @ X.T @ np.diag(W) @ Y
    
    print(H)
    
    return la.svd(H)

def getWeightedValues(state, states, theta, d):
    # calculate weights for each element
    return np.exp(-1 * theta * la.norm(states-state,axis=1) / d)
    """
    weights = np.zeros(states.shape[0])
    current = np.array(state)
    for i, elem in enumerate(states):
        diff = current - elem
        norm = la.norm(diff)
        exponent = -1 * theta * norm / d
        weights[i] = np.exp(exponent)
    return weights
    """

def isInvertible(M):
    return M.shape[0] == M.shape[1] and la.matrix_rank(M) == M.shape[0]
    
def calculateD(states):
    return np.mean(np.fromfunction(lambda i,j: la.norm(states[i]-states[j]),(states.shape[1],states.shape[1]),dtype=int))

def getHat(M, W, x):
    hat = x @ la.pinv(W@M) @ W

    """
    if (isInvertible((W@M).T @ (W@M))):
        # hat = x @ la.inv(X.T @ W @ X) @ (W@X).T @ W
        hat = x @ la.inv((W@M).T @ (W@M)) @ (W@M).T @ W
    else:
        # print("not invertible")
        hat = x @ la.pinv((W@M).T @ (W@M)) @ (W@M).T @ W
        # hat = x @ la.pinv(X.T @ W @ X) @ (W@X).T @ W
        # U, E, V = la.svd(W @ M, full_matrices=False)
        # hat = xaug @ V.T @ np.diag(np.power(E,-1,where=(E!=0))) @ U.T @ W
        # hat = xaug @ V.T @ np.diag(1/(E+1e-10)) @ U.T @ W
        # params = V.T @ np.diag(1/(E+1e-10)) @ (U.T @ W @ Y)[:E.shape[0]]
    # prediction = xaug @ params
    # return x @ la.pinv((W@M).T) @ Y
    """
    return hat

def poincare3d(timeseries, step=1, title="", scatter=True):
    eeee, yyyy = delayEmbed(timeseries, 0, 3, step)
    figPP = plt.figure()
    axPP = figPP.gca(projection="3d")
    if scatter:
        axPP.scatter(eeee[:,0],eeee[:,1],eeee[:,2],linewidth=1)
    else: 
        axPP.plot(eeee[:,0],eeee[:,1],eeee[:,2],linewidth=1)
    axPP.set_xlabel("x(t)")
    axPP.set_ylabel(f"x(t+{step})")
    axPP.set_zlabel(f"x(t+{2*step})")
    axPP.set_title(title)
    axPP.set_xticks([])
    axPP.set_yticks([])
    axPP.set_zticks([])
    plt.show()
    
def poincare2d(timeseries, step=1):
    figTT, axTT = plt.subplots(1)
    axTT.scatter(timeseries[:-step], timeseries[step:])
    axTT.set_xlabel("x(t)")
    axTT.set_ylabel("x(t+1)")
    plt.show()
    
def poincareT(timeseries,step=1,xlabel="x(t)",zlabel="x(t+tau)", scatter=True):
    time = np.linspace(0,1,timeseries.shape[0]-step)
    
    figPP = plt.figure()
    axPP = figPP.gca(projection="3d")
    if scatter:
        axPP.scatter(timeseries[:-step], time, timeseries[step:],linewidth=1)
    else:
        axPP.plot(timeseries[:-step,0], time, timeseries[step:,0],linewidth=1)
    axPP.set_xlabel(xlabel)
    axPP.set_ylabel("t")
    axPP.set_zlabel(zlabel)
    plt.show()

def plotTS(timeseries, title=""):
    figPP, axPP = plt.subplots(1)
    axPP.plot(timeseries)
    axPP.set_xlabel("t")
    axPP.set_ylabel("pop")
    axPP.set_title(title)
    plt.show()

def peakToPeakInterval(X, t, a,b,c):
    imax0 = Xr[a:b].argmax() + a
    imax1 = Xr[b:c].argmax() + b
    return t[imax1] - t[imax0]

def AkaikeTest(AICS, AICNS):
    p1 = np.exp((AICNS-AICS)/2)
    p2 = np.exp((AICS-AICNS)/2)

    if p1 < p2:
        print("Probability SMap beats NSMap: ", p1)
    else:
        print("Probability NSMap beats SMap: ", p2)
    
def likelihoodRatioTest(X, Y, tx, thetaBestS, thetaBest, deltaBest, errThetaDelta):
    nTrials = int(X.shape[0] / 4)
    
    dofS = dofestimation(X, Y, tx, thetaBestS, 0)
    dofG = dofestimation(X, Y, tx, thetaBest, deltaBest)
    dof = abs(dofS - dofG)
    
    teststat = X.shape[0] * np.log(np.min(errThetaDelta[:,0]) / np.min(errThetaDelta))

    print("Probabiliy of SMap superiority : ",chisig(teststat, dof))
    print(f"LambdaLR = ",teststat," dof = ", dof)
    
    return (teststat, dof)
    
    # errS, varS = NSMapMinError(X, t, predHorizon, thetaVals, np.array([0]), nTrials)
    # errG, varG = NSMapMinError(X, t, predHorizon, thetaVals, deltaVals, nTrials)
    
    # return (errS / varS) - (errG / varG)

"""
def likelihoodRatioTest(err1, err2, dof, N):
    lambdaLR = N * np.log(err1 / err2)
    
    if dof == 0:
        return 1
    return 1 - stats.chi2.cdf(lambdaLR,dof)
"""

# WRONG, NEED TO USE APPROPRIATE HAT MATRIX, WHICH IS MADE OF 
def dofestimation(X, Y, tx, theta, delta):

    #_, hat = leaveOneOut(X, Y, tx, theta, delta,get_hat=True)
    # print(hat.shape)
    #dofest = np.trace(hat.T @ hat)
    
    dofest = 0
    for i in range(X.shape[0]):
        pred, hatvector = NSMap(X, Y, tx, X[i], tx[i], theta, delta, return_hat=True)
        dofest += hatvector[0,i]
    return dofest

def chisig(lambdaLR, dof):
    if dof == 0:
        return 1
    return 1 - stats.chi2.cdf(lambdaLR,dof)

# leaves one input and output pair out, and use rest as training data
def leaveOneOut(X, Y, tx, theta, delta, get_hat=False):
    
    if get_hat:
        hat = np.zeros((X.shape[0], X.shape[0]-1))
    timestepPredictions = np.zeros((X.shape[0], 1))
    
    for i in range(0, X.shape[0]):
        # create the train and test stuff
        
        Xjts = X[i].copy()
        Yjts = Y[i].copy()
        tXjts = tx[i].copy()
        
        Xjtr = np.delete(X, i, axis=0)
        Yjtr = np.delete(Y, i, axis=0)
        tXjtr = np.delete(tx, i, axis=0)
        
        if get_hat:
            prediction, hat_vector = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=True)
            hat[i,:] = hat_vector
        else:
            # prediction = NSMap(X, Y, T, x, t, theta, delta, return_hat=False)
            
            # if delta == 1:
            #     # prediction = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=False)
            #     prediction = SMap(Xjtr, Yjtr, Xjts, theta)
            #     # assert prediction1 == prediction
            # else:
            # if delta == 0:
            #    prediction = SMap(Xjtr, Yjtr, Xjts, theta)
            #else:
            prediction = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=False)
        
        timestepPredictions[i] = prediction
            
    if get_hat:
        return (timestepPredictions, hat)
    else:
        return timestepPredictions

def sequential(X, Y, tx, theta, delta, return_error=True):
    trainSize = int(X.shape[0] / 2)
    testSize = X.shape[0] - trainSize
    timestepPredictions = np.zeros((testSize, 1))
    
    for i in range(int(X.shape[0]/2), X.shape[0]):
        # create the train and test stuff        
        # if delta == 0:
        #     prediction = SMap(X[:i], Y[:i], X[i], theta)
        # else:
        prediction = NSMap(X[:i], Y[:i], tx[:i], X[i], tx[i], theta, delta)
        
        timestepPredictions[i - trainSize] = prediction

    if return_error:
        return np.mean((timestepPredictions-Y[trainSize:])**2)
    else:
        return timestepPredictions

def logLikelihood(X, Y, tx, theta, delta, returnSeries=False):
    
    n = Y.shape[0]

    Yhat = leaveOneOut(X, Y, tx, theta, delta)
    k = dofestimation(X, Y, tx, theta, delta)

    mean_squared_residuals = np.sum((Y-Yhat)**2) / (n-k)
    
    lnL = (-n/2)*(np.log(mean_squared_residuals) + np.log(2*np.pi) + 1 )

    if returnSeries:
        return (lnL, Yhat)
    else:
        return lnL

def logUnLikelihood(X, Y, tx, theta, delta, returnSeries=False):
    return -logLikelihood(X, Y, tx, theta, delta, returnSeries=False)

def AIC(X, Y, tx, theta, delta):
    n = X.shape[0]

    lnL = logLikelihood(X, Y, tx, theta, delta)

    k = dofestimation(X, Y, tx, theta, delta)

    AIC = 2 * ( k - lnL )

    return AIC

# leaves one input and output pair out, and use rest as training data
def schreiberContinuous(X, Y, tx, theta, delta):

    n = X.shape[0]
    
    error_matrix = np.zeros((n,n))
    
    for i in range(0, n):
        # create the train and test stuff
        
        Xjts = X[i].copy()
        Yjts = Y[i].copy()
        tXjts = tx[i].copy()
        
        Xjtr = np.delete(X, i, axis=0)
        Yjtr = np.delete(Y, i, axis=0)
        tXjtr = np.delete(tx, i, axis=0)

        for j in range(n-1):
            prediction = NSMap(Xjtr, Yjtr, tXjtr, Xjts, tXjtr[j], theta, delta, return_hat=False)
            error_matrix[i,j] = (Yjts - prediction) ** 2
        
        
    return error_matrix

# make a 1 time step prediction based on a given state(nD vector)
def SMap(X, Y, x, theta):
    norms = la.norm(X-x,axis=1)
    d = np.mean(norms) # d = np.mean(norms) # 
    
    W = np.diag(np.exp(-1 * theta * norms / d))

    H = getHat(X, W, x)
    return H @ Y

    # print(X.shape, np.diag(W).shape, Y.shape)
    # H = la.inv(np.transpose(X) @ np.diag(W) @ X) @ np.transpose(X) @ np.diag(W) @ Y
    # return x @ H

### TIME IS NOT INCLUDED AS A STATE VARIABLE ###

def NSMap(X, Y, T, x, t, theta, delta, return_hat=False):
    # create weights
    norms = la.norm(X - x,axis=1)
    d = np.mean(norms)
    
    tr = (t - np.min(T)) / np.ptp(T)
    Tr = (T - np.min(T)) / np.ptp(T)
    
    weights = np.exp(-1*(theta*norms)/d - delta*(Tr-tr)**2)
    W = np.diag(weights)
    
    Tr = Tr.reshape((T.shape[0],1))
    
    M = np.hstack([X, np.ones(Tr.shape)])
    
    xaug = np.hstack([x, 1])
    xaug = np.reshape(xaug, (1,xaug.shape[0]))
    
    H = getHat(M, W, xaug)
    prediction = H @ Y
    
    if return_hat:
        return (prediction, H)
    else:
        return prediction

"""
### THIS VERSION INCLUDES THE MONOTONICALLY INCREASING DRIVER ###
def NSMap(X, Y, T, x, t, theta, delta, return_hat=False):
    # create weights
    norms = la.norm(X - x,axis=1)
    d = np.mean(norms)
    
    tr = (t - np.min(T)) / np.ptp(T)
    Tr = (T - np.min(T)) / np.ptp(T)
    
    weights = np.exp(-1*(theta*norms)/d - delta*(Tr-tr)**2)
    # weights = np.power(1-delta, abs(Tr-tr)) * np.exp(-1*theta*norms/d)
    # W = np.diag(np.sqrt(weights))
    W = np.diag(weights)
    # weights = np.reshape(weights,(weights.shape[0],1))
    
    Tr = Tr.reshape((T.shape[0],1))
    
    if (delta > 0):
        M = np.hstack([X, Tr, np.ones(Tr.shape)])
        xaug = np.hstack([x, tr, 1])
    else:
        M = np.hstack([X, np.ones(Tr.shape)])
        xaug = np.hstack([x, 1])
    xaug = np.reshape(xaug, (1,xaug.shape[0]))
    
    # H = xaug @ la.pinv((W@M).T @ (W@M)) @ (W@M).T @ W
    # H = xaug @ la.pinv((W@M).T) @ Y
    H = getHat(M, W, xaug)
    prediction = H @ Y
    
    if return_hat:
        return (prediction, H)
    else:
        return prediction
"""
"""
def NSMap(X, Y, T, x, t, theta, delta, return_hat=False):
    # create weights
    norms = la.norm(X - x,axis=1)
    d = np.mean(norms)
    
    tr = t / np.ptp(T)
    Tr = T / np.ptp(T)
    
    weights = np.exp(-1*theta*norms/d - delta*abs(Tr-tr))
    # W = np.diag(np.sqrt(weights))
    W = np.diag(weights)
    # weights = np.reshape(weights,(weights.shape[0],1))
    
    Tr = Tr.reshape((T.shape[0],1))
    
    if (delta > 0):
        M = np.hstack([np.ones(Tr.shape), X, Tr])
        xaug = np.hstack([1, x, tr])
    else:
        M = np.hstack([np.ones(Tr.shape), X])
        xaug = np.hstack([1, x])
    xaug = np.reshape(xaug, (1,xaug.shape[0]))
    
    # hat = xaug @ la.pinv((W@M).T @ (W@M)) @ (W@M).T @ W
    # hat = xaug @ la.pinv((W@M).T) @ Y
    hat = getHat(M, W, xaug)
    prediction = hat @ Y
    
    if return_hat:
        return (prediction, hat)
    else:
        return prediction
"""

def SMapOptimize(Xr, t, horizon, maxLags, stepsize, thetas, returnLandscape=False, minLags=0):
    errorLandscape = np.ones((thetas.shape[0], maxLags+1-minLags))

    for lags in range(minLags,maxLags+1):
        X, Y, tx = delayEmbed(Xr, horizon, lags, stepsize, t=t)
        
        for thetaexp in range(thetas.shape[0]):
            theta = thetas[thetaexp]
            print(f"({theta},{lags+2})")
            
            timestepPredictions = leaveOneOut(X, Y, tx, theta, 0)
            
            totalError = np.sum(abs(timestepPredictions - Y))
            
            errorLandscape[thetaexp, lags-minLags] = totalError
            # print(f"Theta = {theta} Delta = {delta} Error = {errThetaDeltaNSMap[thetaexp, deltaexp]}")

    minError = np.amin(errorLandscape)
    thetaI, lagBest = np.where(errorLandscape == minError)
    # plotOptimization(thetaVals, deltaVals, errorLandscape)

    thetaBest = thetas[thetaI[0]]
    lagBest = lagBest[0]+lagMin

    if returnLandscape:
        return (thetaBest, lagBest, minError, errorLandscape)
    else: 
        return (thetaBest, lagBest, minError)

def NSMapOptimize(Xr, t, horizon, maxLags, stepsize, thetas, deltas, returnLandscape=False, minLags=0):   
    errorLandscape = np.ones((thetas.shape[0], deltas.shape[0], maxLags+1-minLags))

    for lags in range(minLags,maxLags+1):
        X, Y, tx = delayEmbed(Xr, horizon, lags, stepsize, t=t)
        
        for deltaexp in range(deltas.shape[0]):
            for thetaexp in range(thetas.shape[0]):
                theta = thetas[thetaexp]
                delta = deltas[deltaexp]
                print(f"({theta},{delta},{lags+2})")
                
                timestepPredictions = leaveOneOut(X, Y, tx, theta, delta)
                
                totalError = np.sum(abs(timestepPredictions - Y))
                
                errorLandscape[thetaexp, deltaexp, lags-minLags] = totalError
                # print(f"Theta = {theta} Delta = {delta} Error = {errThetaDeltaNSMap[thetaexp, deltaexp]}")

    minError = np.amin(errorLandscape)
    thetaI, deltaI, lagBest = np.where(errorLandscape == minError)
    # plotOptimization(thetaVals, deltaVals, errorLandscape)

    thetaBest = thetas[thetaI[0]]
    deltaBest = deltas[deltaI[0]]
    lagBest = lagBest[0]+lagMin

    if returnLandscape:
        return (thetaBest, deltaBest, lagBest, minError, errorLandscape)
    else: 
        return (thetaBest, deltaBest, lagBest, minError)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def optimizationSuite(Xr, t, horizon, maxLags, errFunc=logUnLikelihood, trainingSteps=30, hp=np.array([0.0,0.0])):

    tableNS = np.zeros(5)
    tableS = np.zeros(4)

    Xemb, Y, tx = delayEmbed(Xr, horizon, maxLags, 1, t=t)

    # for each number of lags from 0 to maxLags
    for tau in range(1,1+int((maxLags+1)/2)):
        for l in range(maxLags+2):
            if (tau > 1 and l == 0) or ((l+1)*tau >= Xemb.shape[1]):
                continue
            
            print(f"E = {l+2}, tau = {tau}")

            X = Xemb[:,:(l+1)*tau:tau]
            #else:
            #   emb_array = (np.ones(Xr.shape[1])*l).astype(int)
            #   X, Y = delayEmbedM(Xr[:-horizon], Xr[horizon:,0,None], emb_array, lagStepsize)
            #   tx = np.linspace(0,1,num=X.shape[0])

            print("NSMap")
            thetaNS, deltaNS, errNS = optimizeG(X, Y, tx, errFunc=errFunc, hp=hp.copy())
            print("SMap")
            thetaS, _, errS = optimizeG(X, Y, tx, errFunc=errFunc, hp=hp.copy(), fixed=np.array([False, True]))

            tableNS = np.vstack([tableNS, np.array([errNS, thetaNS, deltaNS, l, tau])])
            tableS = np.vstack([tableS, np.array([errS, thetaS, l, tau])])

    tableNS = np.delete(tableNS, 0, 0)
    tableS = np.delete(tableS, 0, 0)

    iNS = np.argmin(tableNS[:,0])
    iS = np.argmin(tableS[:,0])
    
    # return best hyperparameters and minimum error for each
    print(f"NSMap: \n Max Likelihood {-tableNS[iNS][0]} \n E: {2+int(tableNS[iNS][3])} \n tau: {int(tableNS[iNS][4])} \n Theta: {tableNS[iNS][1]} \n Delta: {tableNS[iNS][2]}")
    print(f"SMap: \n Max Likelihood {-tableS[iS][0]} \n E: {2+int(tableS[iS][2])} \n tau: {int(tableS[iS][3])} \n Theta: {tableS[iS][1]}")
    
    # (thetaNS, deltaNS, errNS, lagsNS, tauNS, thetaS, errS, lagsS, tauS)
    return (tableNS[iNS][1], tableNS[iNS][2], -tableNS[iNS][0], int(tableNS[iNS][3]), int(tableNS[iNS][4]), tableS[iS][1], -tableS[iS][0], int(tableS[iS][2]), int(tableS[iS][3]))


# numerically evaluates gradient
def gradient(X, Y, tx, theta, delta, errFunc=logUnLikelihood):
    # this is the gap used to evaluate the gradient
    D = 0.01
    
    ddelta = max(min(D, (1-delta)/2), 10e-5)
    dtheta = D

    E = errFunc(X, Y, tx, theta, delta)
    
    Et= errFunc(X, Y, tx, theta + dtheta, delta)
    
    Ed = errFunc(X, Y, tx, theta, delta + ddelta)

    grad = ((E-Et)/dtheta, (E-Ed)/ddelta)

    return (grad, E)

"""
# Optimize SMap using GRADIENT DESCENT instead of evaluating a grid
def SMapOptimizeG(X, Y, t, errFunc=leaveOneOut, trainingSteps=20, thetaInit=0):

    err = 0
    count = 0

    rhoplus = 1.1 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1
    
    hp = np.array([thetaInit], dtype=float)
    gradPrev = np.array([1], dtype=float)
    deltaPrev = np.array([1], dtype=float)
    
    while la.norm(gradPrev) > 0.001 and count < trainingSteps:
        grad, err = gradient(X, Y, t, hp[0], 0, errFunc=errFunc)
        grad = grad / la.norm(grad)# np.abs(grad) # NORMALIZE, because rprop ignores magnitude

        s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
        spos = np.ceil(s) # 0 for - vals, 1 for + vals
        sneg = -1 * (spos - 1)

        delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
        dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

        deltaPrev = delta
        gradPrev = grad
        count += 1

        # floor and ceiling on the hyperparameters
        hp[0] = max(0, hp[0] + dweights[0])

        print(hp)
        print(err)
    return (hp[0], err)
"""
# Optimize using GRADIENT DESCENT instead of evaluating a grid
def optimizeG(X, Y, t, errFunc=logUnLikelihood, trainingSteps=20, hp=np.array([0.0,0.0]), fixed=np.array([False, False])):    
    err = 0
    
    gradPrev = np.ones(hp.shape, dtype=float)
    deltaPrev = np.ones(hp.shape, dtype=float)
    
    for count in range(trainingSteps):
        errPrev = err
        
        grad, err = gradient(X, Y, t, hp[0], hp[1], errFunc=errFunc)

        print(f"[{count+1:02d}] theta: {hp[0]:.3f}, delta: {hp[1]:.3f}, log Likelihood: {-err:.3f}")

        if abs(err-errPrev) < 0.01 or count == trainingSteps-1:
            break

        dweights, deltaPrev, gradPrev = calculateHPChange(grad, gradPrev, deltaPrev)
        
        # floor and ceiling on the hyperparameters
        for i in range(2):
            if not fixed[i]:
                hp[i] = max(0, hp[i] + dweights[i])


    return (hp[0], hp[1], err)

def calculateHPChange(grad, gradPrev, deltaPrev):
    rhoplus = 1.1 # if the sign of the gradient doesn't change, must be > 1
    rhominus = 0.5 # if the sign DO change, then use this val, must be < 1
    
    grad = grad / la.norm(grad)# np.abs(grad) # NORMALIZE, because rprop ignores magnitude

    s = np.multiply(grad, gradPrev) # ratio between -1 and 1 for each param
    spos = np.ceil(s) # 0 for - vals, 1 for + vals
    sneg = -1 * (spos - 1)

    delta = np.multiply((rhoplus * spos) + (rhominus * sneg), deltaPrev)
    dweights = np.multiply(delta, ( np.ceil(grad) - 0.5 ) * 2) # make sure signs reflect the orginal gradient

    return (dweights, delta, grad)
                 
"""
def NSMapOptimize(X, Y, tx, thetaVals, deltaVals, calc_hat=False):
    errThetaDelta = np.ones((thetaVals.shape[0], deltaVals.shape[0]))

    lowestError = float('inf')

    thetaBest = 0
    deltaBest = 0
    lowestError = float('inf')
    lowestVariance = 0
    for deltaexp in range(deltaVals.shape[0]):
        for thetaexp in range(thetaVals.shape[0]):
            theta = thetaVals[thetaexp]
            delta = deltaVals[deltaexp]
            
            # timestepPredictions = predictionHorizon(X, Y, t, theta, delta, predHorizon, nTrials)
            if calc_hat:
                timestepPredictions, hat = leaveOneOut(X, Y, tx, theta, delta, True)
            else:
                timestepPredictions = leaveOneOut(X, Y, tx, theta, delta)
            
            totalError = np.sum(abs(timestepPredictions - Y))
            if totalError < lowestError:
                lowestError = totalError
                deltaBest = delta
                thetaBest = theta
                lowestVariance = np.var(timestepPredictions)
            
            errThetaDelta[thetaexp, deltaexp] = totalError
            # print(f"Theta = {theta} Delta = {delta} Error = {errThetaDeltaNSMap[thetaexp, deltaexp]}")

    plotOptimization(thetaVals, deltaVals, errThetaDelta)
    
    if (calc_hat):
        return (thetaBest, deltaBest, errThetaDelta, hat)
    else:
        return (thetaBest, deltaBest, errThetaDelta)
"""
                 
def plotOptimization(thetas, deltas, errNSMap):    
    # Theta Optimization

    fig2, ax2 = plt.subplots(1, figsize = (8,8))
    plt.colorbar(ax2.imshow(np.log(errNSMap)))
    ax2.set_xticks(np.arange(deltas.shape[0]))
    ax2.set_xticklabels(list(np.round(deltas,2)))
    ax2.set_xlabel("Delta")
    ax2.set_yticks(np.arange(thetas.shape[0]))
    ax2.set_yticklabels(list(np.round(thetas,2)))
    ax2.set_ylabel("Theta")
    ax2.set_title("NSMap Error Results")
    plt.show()

    print(f"Min SMap Error: {np.min(errNSMap[:,0])}, Min NSMap Error: {np.min(errNSMap)}")
    print(f"Improvement of NSMap: {np.min(errNSMap[:,0])/np.min(errNSMap)}")

def functionSurfaceSMap(Xr, predHorizon, theta, resolution):
    X, Y = delayEmbed(Xr, predHorizon, 1, 1, removeNAs=True)  
    U = np.max(Y)
    L = np.min(Y)
    
    # Create Function Surface
    r = np.linspace(L, U, num=resolution)
    A, B = np.meshgrid(r, r)
    N = np.zeros(resolution)
    C = np.zeros((resolution,resolution))
    for i in range(resolution):
        for j in range(resolution):
            x = np.array([A[i,j], B[i,j]])
            C[i,j] = NSMap(X, Y, np.linspace(0,1, X.shape[0]), x, 0, theta, 0)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(A, B, C,color="silver")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t-1)")
    ax.set_zlabel("NSMap(x(t),x(t-1))")
    ax.scatter(X[:,0], X[:,1], Y)
    
    plt.show()

def functionSurfaceNSMap(Xr, predHorizon, theta, delta, resolution):
    X, Y = delayEmbed(Xr, predHorizon, 0, 1, removeNAs=True)
    U = np.max(Y)
    L = np.min(Y)
    
    # Create Function Surface
    r = np.linspace(L, U, num=resolution)
    T = np.linspace(0,1,num=resolution)
    A, B = np.meshgrid(r,T)
    C = np.zeros((resolution,resolution))
    for i in range(resolution):
        for j in range(resolution):
            C[i,j] = NSMap(X, Y, np.linspace(0,1, X.shape[0]), A[i,j], B[i,j], theta, delta)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(A, B, C, color="green")
    ax.set_xlabel("x(t)")
    ax.set_ylabel("t")
    ax.set_zlabel("NSMap(x(t),t)")
    ax.scatter(X, np.linspace(0,1, X.shape[0]), Y)

    ax.axes.set_zlim3d(bottom=np.min(Xr),top=np.max(Xr))

    plt.show()

def MDStimescaleDecomp(Xr, lags=3, window_size=0.2):
    X, _ = delayEmbed(Xr, 1, lags, 1)
    
    n = X.shape[0]
    
    distance_matrix = distanceMatrix(X)

    similarity_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i,j] = dynamicSimilarity(distance_matrix, i, j, window_size=window_size)
    print(f"{i},{j}:{similarity_matrix[i,j]}")
    
    similarity_matrix = standardize(similarity_matrix)
    
    fig, ax = plt.subplots(1)
    ax.imshow(similarity_matrix)
    plt.show()
    
    embedding = MDS(dissimilarity="precomputed")
    X_transformed = embedding.fit_transform(similarity_matrix)
    
    return X_transformed
    
"""
def dynamicSimilarity(distance_matrix, t1, t2, window_size=0.2):
    n = distance_matrix.shape[0]
    n_neighbors = 3
    
    win_radius = int(n * window_size)
    
    U = max(0, t1-win_radius)
    D = min(n, t1+win_radius)
    L = max(0, t2-win_radius)
    R = min(n, t2+win_radius)
    
    window = distance_matrix[U:D, L:R]

    distance = np.mean(np.exp(-window))
    
    return distance

    dynamic_similarity = 0
    for x1 in window1:
        neighbors = nearestNeighbors(x1, window2, n_neighbors)
        for neighbor in neighbors:
            dynamic_similarity += la.norm(neighbor-x1)**2
    
    return dynamic_similarity / (n*n_neighbors)
    
"""

"""
def driverVdelta(resolution):
    # Final data will be
    # Nonstat Rate(0,1)|thetaNS|deltaNS|errNS(l1o)|errNS(seq)|lagNS|dofNS|thetaS|errS(l1o)|errS(seq)|lagS|dofS
    
    table = np.zeros((resolution, 12))
    
    x0 = np.array([0.1,0.4,9])
    for r in range(resolution):
        rate = float(r)/resolution
        b1 = lambda t: 2.5 + rate * 4 * t / end
        
        Xr = standardize(generateTimeSeriesContinuous('HastingsPowellP', x0, nsargs=(b1,), end=end, tlen = tlen, reduction = reduction, settlingTime=settlingTime))[:,0,None]

        predictionHorizon = 1
        lagStep = 1
        maxLags = 6
        
        plotTS(Xr)
        
        thetaNS, deltaNS, errNS, lagsNS, thetaS, errS, lagsS = optimizationSuite(Xr, tr, predictionHorizon, maxLags, lagStep)
        
        Xn, Yn, txn = delayEmbed(Xr, predictionHorizon, lagsNS, lagStep, t=t)
        dofNS = dofestimation(Xn, Yn, txn, thetaNS, deltaNS)

        Xs, Ys, txs = delayEmbed(Xr, predictionHorizon, lagsS, lagStep, t=t)
        dofS = dofestimation(Xs, Ys, txs, thetaNS, 0)
        
        MSENS, sequentialNS = sequential(Xn, Yn, txn, thetaNS, deltaNS, returnSeries=True)
        MSES, sequentialS = sequential(Xs, Ys, txs, thetaS, 0, returnSeries=True)
        
        stinky = np.array([rate,thetaNS,deltaNS,errNS,MSENS,lagsNS,dofNS,thetaS,errS,MSES,lagsS,dofS])
        for pp in range(12):
            table[r,pp] = stinky[pp]
            
        AkaikeTest(errNS, errS, dofNS, dofS, Xr.shape[0])
        
    return table
"""   
