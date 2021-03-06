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
testvtrainratio = .9

# Read input data from files
file = "ticks-targets - Cleaned.csv"
data = pd.read_csv(file,encoding="utf-8",na_filter=False)
print(data)

"""
x = list(map(lambda elem: elem[0], states))
y = list(map(lambda elem: elem[1], states))
z = list(map(lambda elem: elem[2], states))
"""

# states contains a numpy array of lorenz equations time series data
states = data.to_numpy(dtype=np.float32)
print(states)

# data prep

states[:,0] %= 100
newCol = np.cos(states[:,0] * 2.0 * np.pi / 52.0)
states[:,0] = np.sin(states[:,0] * 2.0 * np.pi / 52.0)
print(states)

tao = 13
colNum = 0
print("STD ", states[:3,0]," ", np.mean(states, axis=0, dtype=np.float64)[0])
# states[:,0] = states[:,0] / np.std(states, axis=0, dtype=np.float64)[0]
# states[:,0] = states[:,0] - np.mean(states, axis=0, dtype=np.float64)[0]
states[:,1] = states[:,1] / np.std(states, axis=0, dtype=np.float64)[1]
states[:,1] = states[:,1] - np.mean(states, axis=0, dtype=np.float64)[1]

print("States ",states)
# newCol = states[:-tao,colNum]
print("NewCol ",newCol," ", newCol.size)
# states = states[tao:,]
# np.resize(states, (2, states.shape[0] - tao))
print("States ",states)
states = np.column_stack((states, newCol))
# np.append(states, newCol, axis=0)
print(states)


i1 = np.array([3000,2000])

# prediction!
megaPrediction = []
numIterations = 30
prevState = states[-1]
for i in range(20):
    prevState = SMapPrediction(prevState,states)
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
#plt.plot(data["paramecium"],data["didinium"])
plt.plot(states[:,0], states[:,1], states[:,2])

# Single Step Prediction Accuracy
errors = []
for i in range(len(states)-1):
    prediction = SMapPrediction(states[i], states)
    errors.append(la.norm(states[i+1] - prediction))

errorfig = plt.figure(2)
errorAx = errorfig.add_subplot()
errorAx.hist(errors, bins = 10)
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
