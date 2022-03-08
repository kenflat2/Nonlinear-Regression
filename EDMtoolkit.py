import numpy as np
import math
import numpy.linalg as la
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epsilon = 2e-10

def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def nearestNeighbors(s0, S, n):
    orderedNeighbors = np.argsort(la.norm(s0 - S[:-1],axis=1))    
    return orderedNeighbors[1:n+1]

# create a delay embeddding vector from a given UNIVARIATE time series.
def delayEmbed(Dr, predHorizon, nLags, embInterval, t = None, removeNAs=False):
    # Remove NAs before embedding
    if removeNAs:
        notNA = np.all(~np.isnan(Dr),axis=1)

        D = Dr[notNA]
        if t is not None:
            t = t[notNA]
    else:
        D = Dr
    
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
    else: 
        B = A
    
    if t is None:
        return (B[:,1:], B[:,0, None])
    else:
        ty = t[predHorizon : predHorizon - rowsLost]
        tx = t[:-rowsLost]
        return (B[:,1:], B[:,0, None], tx, ty)

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
            for i in range(l+1):
                Y, _ = delayEmbed(Xr[:,d,None], 0, i, s) # individual axis version
                # Y, _ = delayEmbed(Xr[::c], Xr[::c], [i]*dim,s)
                # Y, _ = delayEmbedUnitary(Xr[::c], Xr[::c], i,s)
                lyapExps[i] = lyapunovExp(Y)
            # print(lyapExps)
            if dim == 1:
                axFNN.plot(range(l+1), lyapExps, label="{e}".format(e=s))
            else:
                axFNN[d].plot(range(l+1), lyapExps, label="{e}".format(e=s))
        
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

def poincare3d(timeseries):
    eeee, yyyy = delayEmbed(timeseries, 0, 3, 1)
    figPP = plt.figure()
    axPP = figPP.gca(projection="3d")
    axPP.plot(eeee[:,0],eeee[:,1],eeee[:,2],linewidth=1)
    axPP.set_xlabel("x(t)")
    axPP.set_ylabel("x(t+1)")
    axPP.set_zlabel("x(t+2)")
    plt.show()
    
def poincare2d(timeseries):
    figTT, axTT = plt.subplots(1)
    axTT.plot(timeseries[:-1], timeseries[1:])
    axTT.set_xlabel("x(t)")
    axTT.set_ylabel("x(t+1)")
    plt.show()
    
def poincareT(timeseries):
    figPP = plt.figure()
    axPP = figPP.gca(projection="3d")
    axPP.scatter(timeseries[:-1],timeseries[1:],np.linspace(0,1,timeseries.shape[0]-1),linewidth=1)
    axPP.set_xlabel("x(t)")
    axPP.set_ylabel("x(t+1)")
    axPP.set_zlabel("t")
    plt.show()

def plotTS(timeseries):
    figPP, axPP = plt.subplots(1)
    axPP.plot(timeseries)
    axPP.set_xlabel("t")
    axPP.set_ylabel("pop")
    plt.show()

def peakToPeakInterval(X, t, a,b,c):
    imax0 = Xr[a:b].argmax() + a
    imax1 = Xr[b:c].argmax() + b
    return t[imax1] - t[imax0]


def likelihoodRatioTest(X, Y, tx, ty, thetaBestS, thetaBest, deltaBest, errThetaDelta):
    nTrials = int(X.shape[0] / 4)
    
    dofS = dofestimation(X, Y, tx, thetaBestS, 0)
    dofG = dofestimation(X, Y, tx, thetaBest, deltaBest)
    dof = abs(dofS - dofG)
    
    teststat = X.shape[0] * np.log(np.min(errThetaDelta[:,0]) / np.min(errThetaDelta))
    
    return (teststat, dof)
    
    # errS, varS = GMapMinError(X, t, predHorizon, thetaVals, np.array([0]), nTrials)
    # errG, varG = GMapMinError(X, t, predHorizon, thetaVals, deltaVals, nTrials)
    
    # return (errS / varS) - (errG / varG)

def likelihoodRatioTest(err1, err2, dof, N):
    lambdaLR = N * np.log(err1 / err2)
    
    if dof == 0:
        return 1
    return 1 - stats.chi2.cdf(lambdaLR,dof)
    
# WRONG, NEED TO USE APPROPRIATE HAT MATRIX, WHICH IS MADE OF 
def dofestimation(X, Y, tx, theta, delta):
    dofest = 0
    for i in range(X.shape[0]):
        pred, hatvector = GMap(X, Y, tx, X[i], tx[i], theta, delta, return_hat=True)
        dofest += hatvector[0,i]
    return dofest
        
def chisig(test_stat, dof):
    if dof == 0:
        return 1
    return 1 - stats.chi2.cdf(lambdaLR,dof)

# leaves one input and output pair out, and use rest as training data
def leaveOneOut(X, Y, tx, theta, delta, get_hat=False):
    
    if get_hat:
        hat = np.zeros((X.shape[0]-1, X.shape[0]-1))
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
            prediction, hat_vector = GMap(X, Y, tx, Xjts, tXjts, theta, delta, return_hat=True)
            if i < X.shape[0]-1:
                hat[i,:] = hat_vector
        else:
            # prediction = GMap(X, Y, T, x, t, theta, delta, return_hat=False)
            
            if delta == 0:
                # prediction = GMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=False)
                prediction = SMap(Xjtr, Yjtr, Xjts, theta)
                # assert prediction1 == prediction
            else:
                prediction = GMap(Xjtr, Yjtr, tXjtr, Xjts, tXjts, theta, delta, return_hat=False)
        
        timestepPredictions[i] = prediction
            
    if get_hat:
        return (timestepPredictions, hat)
    else:
        return timestepPredictions

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

def GMap(X, Y, T, x, t, theta, delta, return_hat=False):
    # create weights
    norms = la.norm(X - x,axis=1)
    d = np.mean(norms) + delta
    
    tr = t / np.ptp(T)
    Tr = T / np.ptp(T)
    
    weights = np.exp(-1*(theta*norms + delta*abs(Tr-tr))/d)
    # W = np.diag(np.sqrt(weights))
    W = np.diag(weights)
    # weights = np.reshape(weights,(weights.shape[0],1))
    
    Tr = Tr.reshape((T.shape[0],1))
    
    if (delta > 0):
        M = np.hstack([X, Tr])
        xaug = np.hstack([x, tr])
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

def GMapOptimize(X, Y, tx, thetaVals, deltaVals, calc_hat=False):
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
            
            totalError = np.sum((timestepPredictions - Y)**2)
            if totalError < lowestError:
                lowestError = totalError
                deltaBest = delta
                thetaBest = theta
                lowestVariance = np.var(timestepPredictions)
            
            errThetaDelta[thetaexp, deltaexp] = totalError
            # print(f"Theta = {theta} Delta = {delta} Error = {errThetaDeltaGMap[thetaexp, deltaexp]}")

    plotOptimization(thetaVals, deltaVals, errThetaDelta)
    
    if (calc_hat):
        return (thetaBest, deltaBest, errThetaDelta, hat)
    else:
        return (thetaBest, deltaBest, errThetaDelta)


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
    ax2.set_title("GMap Error Results")
    plt.show()

    print(f"Min SMap Error: {np.min(errNSMap[:,0])}, Min GMap Error: {np.min(errNSMap)}")
    print(f"Improvement of GMap: {np.min(errNSMap[:,0])/np.min(errNSMap)}")
