import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

theta = 6
dim = 3

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

numNeighbors = 3

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
def SMapPrediction(state,states):
    W = getWeightedValues(state, states)
    W = np.delete(W, (-1), axis = 0)
    X = np.delete(states, (-1), axis=0)
    Y = np.delete(states, (0), axis=0)
    H = la.inv(np.transpose(X) @ np.diag(W) @ X) @ np.transpose(X) @ np.diag(W) @ Y
    # print("State ", state, "H ",H, "Prediction ", state @ H)
    return state @ H
    """
    b = a[:,0]
    lastElem = b[len(b)-1]
    for i in range(1,len(b)):
        b[i] = b[i-1]
    b[0] = lastElem
    print("a ",a)
    print("b ",b)
    
    return la.lstsq(a,b)
    
    weightedSum = np.array([0.0] * dim)
    for i in range(len(states)-1):
        weightedSum += states[i+1] * weights[i]
    finalPred = weightedSum / len(states)
    return finalPred
    """

def getWeightedValues(state, states):
    d = calculateD(state, states)
    # calculate weights for each element
    weights = []
    current = np.array(state)
    for elem in states:
        diff = current - elem
        norm = la.norm(diff)
        exponent = -1 * theta * norm / d
        weights.append(np.e ** exponent)
    """
    output = np.ndarray(shape=(len(states),len(state)), dtype=float)
    for i in range(len(states)):
        output[i] = states[i] * weights[i]
    print("FINAL OUTPUT ",output)
    """
    return weights

def calculateD(state, states):
    norms = []
    current = np.array(state)
    for elem in states:
        diff = current - elem
        norm = la.norm(diff)
        norms.append(norm)
    d = sum(norms) / len(norms)
    return d

# main logic
start = 0
end = 50
step = .05
testvtrainratio = .9
cutoff = int(end * (1/step) * testvtrainratio)

state0 = np.array([1.0, 2.0, 4.0])
t = np.arange(start, end, step)

# generate input data
states = odeint(f, state0, t)
trainStates = states[:cutoff]
testStates = states[cutoff:]
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
i1 = np.array([4,7,2])
pred1 = nearestNeighborsPrediction(i1)
print(pred1)

pred2 = SMapPrediction(i1,trainStates)
print("SMap Prediction = ", pred2)

megaPrediction = []
numIterations = 30
prevState = trainStates[-1]
for i in range(len(testStates)):
    prevState = SMapPrediction(prevState, trainStates)
    megaPrediction.append(prevState)

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
fig2 = plt.figure(2)
# prediction iterated over time
ax1 = fig2.gca(projection="3d")
ax1.plot(x, y, z, color='tab:red')
ax1.plot(testStates[:,0],testStates[:,1],testStates[:,2],color='tab:blue')
ax1.set_title("Predicted(red) vs Acutal(blue)")

# prediction accuracy over time
fig3 = plt.figure(3)
ax2 = fig3.add_subplot()
ax2.plot(range(len(stateDistance)), stateDistance)

ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")

plt.show()
