import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from numpy import genfromtxt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

theta = 1
dim = 3

numNeighbors = 3

def nearestNeighbors(state, trainStates):
    orderedNeighbors = sorted(range(len(trainStates)-1), key = lambda i : la.norm(state - trainStates[i]), reverse=False)
    return orderedNeighbors[:numNeighbors]

def nearestNeighborsPrediction(state, trainStates):
    neighborIndexes = nearestNeighbors(state, trainStates)
    pred1neigh = list(map(lambda i: trainStates[i+1], neighborIndexes))
    return sum(pred1neigh) / numNeighbors

"""
# make a 1 time step prediction based on a given state(nD vector)
def nextStatePrediction(state):
    d = calculateD(state)
    # calculate weights for each element
    weights = []
    current = np.array(state)
    for elem in states:
        diff = current - elem
        norm = la.norm(diff)
        exponent = -1 * theta * norm / d
        weights.append(np.e ** exponent)
    # print(weights)
    
    weightedSum = np.array([0.0] * dim)
    for i in range(len(states)-1):
        weightedSum += states[i+1] * weights[i]
    finalPred = weightedSum / len(states)
    return finalPred

def calculateD(state):
    norms = []
    current = np.array(state)
    for elem in states:
        diff = current - elem
        norm = la.norm(diff)
        norms.append(norm)
    d = sum(norms) / len(norms)
    return d

def nNextStatePrediction(state, n):
    currentPrediction = state
    states = []
    for i in range(n):
        currentPrediction = nextStatePrediction(currentPrediction)
        states.append(currentPrediction)
    return states
"""

# Read input data from files
file = "lynxhare - cleaned.csv"
data = pd.read_csv(file,encoding="utf-8",dtype={"Lynx Count":np.int32,"Hare Count":np.int32},na_filter=False)
print(data)

# main logic
testvtrainratio = .9


"""
x = list(map(lambda elem: elem[0], states))
y = list(map(lambda elem: elem[1], states))
z = list(map(lambda elem: elem[2], states))
"""

# states contains a numpy array of lorenz equations time series data
states = data.to_numpy()
print(states)
i1 = np.array([3000,2000])
pred1 = nearestNeighborsPrediction(i1,states)
print(pred1)

megaPrediction = []
numIterations = 30
prevState = states[-1]
for i in range(20):
    prevState = nearestNeighborsPrediction(prevState,states)
    megaPrediction.append(prevState)
"""
stateDistance = list(map(lambda pred, actual: la.norm(pred-actual),megaPrediction,testStates))
print(stateDistance)
"""
x = list(map(lambda elem: elem[0], megaPrediction))
y = list(map(lambda elem: elem[1], megaPrediction))
# z = list(map(lambda elem: elem[2], megaPrediction))

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,c='r',marker='o')

# input dynamics
plt.plot(data["Lynx Count"],data["Hare Count"])
"""
# prediction iterated over time
ax1 = fig2.gca(projection="3d")
ax1.plot(x, y, z, color='tab:red')
ax1.plot(testStates[:,0],testStates[:,1],testStates[:,2],color='tab:blue')
ax1.set_title("Predicted(red) vs Acutal(blue)")

# prediction accuracy over time
fig3 = plt.figure(3)
ax2 = fig3.add_subplot()
ax2.plot(range(len(stateDistance)), stateDistance)
"""
#ax.set_xlabel("X label")
#ax.set_ylabel("Y label")
#ax.set_zlabel("Z label")

plt.show()
