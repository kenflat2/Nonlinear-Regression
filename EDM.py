import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from functools import reduce

dim = 3

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

numNeighbors = 3

def Lorenz96P(x, t, F):
    N = 5 # dimension

    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F(t)
    return d   

def f(state,t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def nearestNeighbors(state, n):
    orderedNeighbors = sorted(range(len(trainStates)-1), key = lambda i : la.norm(state - trainStates[i]), reverse=False)
    return orderedNeighbors[:n]

def nearestNeighborsPrediction(state):
    neighborIndexes = nearestNeighbors(state, numNeighbors)
    pred1neigh = list(map(lambda i: trainStates[i+1], neighborIndexes))
    return sum(pred1neigh) / numNeighbors

# make a 1 time step prediction based on a given state(nD vector)
def SMapPrediction(state,states, theta, d):
    W = getWeightedValues(state, states, theta, d)
    W = np.delete(W, (-1), axis = 0)
    X = np.delete(states, (-1), axis=0)
    Y = np.delete(states, (0), axis=0)
    H = la.inv(np.transpose(X) @ np.diag(W) @ X) @ np.transpose(X) @ np.diag(W) @ Y
    # print("State ", state, "H ",H, "Prediction ", state @ H)
    return state @ H

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
    
def calculateD(states):
    return np.mean(np.fromfunction(lambda i,j: la.norm(states[i]-states[j]),(states.shape[1],states.shape[1]),dtype=int))

def getMegaPrediction(num, trainStates, theta, d):
    megaPrediction = []
    prevState = trainStates[-1]
    for i in range(num):
        prevState = SMapPrediction(prevState, trainStates, theta, d)
        megaPrediction.append(prevState)

    return megaPrediction
    x = list(map(lambda elem: elem[0], megaPrediction))
    y = list(map(lambda elem: elem[1], megaPrediction))
    z = list(map(lambda elem: elem[2], megaPrediction))
    
def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

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
            lag = 1
            for _ in range(assignment[i]):
                newCol = Xin[:-embInterval*lag,i]
                Xin[embInterval*lag:, tmplen + newColInd] = newCol
                newColInd += 1
                lag += 1
    Xin = Xin[embInterval*sum(assignment):]
    Yin = Yin[embInterval*sum(assignment):]
    
    # Yin = Yin[-X.shape[0]:]
    
    return (Xin, Yin)

# main logic
start = 0
end = 2**7
tlen = 2**14
reduction = 2 ** 4
testvtrainratio = .95
cutoff = int(tlen * testvtrainratio / reduction)

# state0 = np.array([1.0, 2.0, 4.0])
t0 = np.ones(5)
t0[0] += 0.1
t = np.linspace(start, end, num=tlen)

# generate input data
# states = odeint(f, state0, t)
F = lambda t : 7 + 2 * t / end
states = standardize(odeint(Lorenz96P, t0, t, args=(F,))[::reduction,0,None])
t = t[::reduction]
np.set_printoptions(suppress=True)
print(states, t)
states = np.hstack([states, t.reshape((t.shape[0], 1))]) # this is the GMAP step
print(states)
trainStates = states[:cutoff]
testStates = states[cutoff:]

d = calculateD(states)
print("D = ", d)

# print(trainStates)

# Read input data from files
# file = "lynxhare - cleaned.csv"
# data = pd.read_csv(file,encoding="utf-8",na_filter=False)
# print(data)

"""
x = list(map(lambda elem: elem[0], states))
y = list(map(lambda elem: elem[1], states))
z = list(map(lambda elem: elem[2], states))
"""

# states contains a numpy array of lorenz equations time series data
i1 = np.array([52,33])
pred1 = nearestNeighborsPrediction(i1)
print(pred1)

pred2 = SMapPrediction(i1,trainStates, 4, d)
print("SMap Prediction = ", pred2)

megaPrediction = getMegaPrediction(int(tlen*(1-testvtrainratio)), trainStates, 12, d)
stateDistance = list(map(lambda pred, actual: la.norm(pred-actual),megaPrediction,testStates))
# print(stateDistance)

x = list(map(lambda elem: elem[0], megaPrediction))
y = list(map(lambda elem: elem[1], megaPrediction))
z = list(map(lambda elem: elem[2], megaPrediction))

fig = plt.figure(1)
"""
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c='r',marker='o')
"""
# input dynamics
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_title("Training Data")
fig2 = plt.figure(2)

# prediction iterated over time
ax1 = fig2.gca(projection="3d")
ax1.plot(x, y, z, color='tab:red')
ax1.plot(testStates[:,0],testStates[:,1],testStates[:,2],color='tab:blue')
ax1.set_title("Feed Forward Prediction(red) vs Acutal Test Data(blue)")

# prediction accuracy over time
fig3 = plt.figure(3)
ax2 = fig3.add_subplot()
ax2.plot(range(len(stateDistance)), stateDistance)
ax2.set_ylabel("Error")
ax2.set_xlabel("Time Steps")

# user interaction stuff
fig4 = plt.figure(4)
thetaAx = plt.axes()
thetaChooser = Slider(thetaAx,"Theta", 0, 15, valinit=12, valstep=1)
thetaAx.set_title("High Theta = small neighborhood, low = large neighborhood")

def update(val):
    theta = thetaChooser.val
        
    megaPrediction = getMegaPrediction(int(tlen*(1-testvtrainratio)), trainStates, theta,d)
    stateDistance = list(map(lambda pred, actual: la.norm(pred-actual),megaPrediction,testStates))
    # print(stateDistance)

    x = list(map(lambda elem: elem[0], megaPrediction))
    y = list(map(lambda elem: elem[1], megaPrediction))
    z = list(map(lambda elem: elem[2], megaPrediction))
    
    ax1.clear()
    ax1.plot(x, y, z, color='tab:red')
    ax1.plot(testStates[:,0],testStates[:,1],testStates[:,2],color='tab:blue')
    fig2.canvas.draw()
    fig2.canvas.flush_events()

thetaChooser.on_changed(update)

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")

plt.show()
