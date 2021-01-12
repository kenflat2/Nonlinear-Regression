import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from numpy import genfromtxt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from functools import reduce

theta = 1
dim = 3
tao0 = 20
timeDelayVariable0 = 0

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

# main logic
start = 0
end = 40
step = .01

state0 = np.array([1.0, 2.0, 4.0])
t = np.arange(start, end, step)

# generate input data
states = odeint(f, state0, t)

# Read input data from files
file = "lynxhare - Sheet1.csv"
data = pd.read_csv(file,encoding="utf-8",na_filter=False)
# print(data)

"""
x = list(map(lambda elem: elem[0], states))
y = list(map(lambda elem: elem[1], states))
z = list(map(lambda elem: elem[2], states))
"""

# states contains a numpy array of lorenz equations time series data
"""
i1 = np.array([4,7,2])
pred1 = nearestNeighborsPrediction(i1)
print(pred1)


megaPrediction = []
numIterations = 30
prevState = trainStates[-1]
for i in range(len(testStates)):
    prevState = nearestNeighborsPrediction(prevState)
    megaPrediction.append(prevState)

stateDistance = list(map(lambda pred, actual: la.norm(pred-actual),megaPrediction,testStates))
print(stateDistance)

x = list(map(lambda elem: elem[0], megaPrediction))
y = list(map(lambda elem: elem[1], megaPrediction))
z = list(map(lambda elem: elem[2], megaPrediction))

fig = plt.figure(1)

ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,c='r',marker='o')
"""

# time delay embeddingx
delay1 = states[:-tao0*2,timeDelayVariable0]
delay2 = states[tao0:-tao0,timeDelayVariable0]
delay3 = states[tao0*2:,timeDelayVariable0]
print(delay1)
print(delay2)
print(delay3)

# raw input
fig = plt.figure(1)
ax = fig.gca(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])

# time delay
fig2 = plt.figure(2)
ax1 = fig2.gca(projection="3d")
ax1.plot(delay1, delay2, delay3)

# user interaction stuff
fig3 = plt.figure(3)
taoAx = plt.axes()
taoChooser = Slider(taoAx,"Tao", 0, 100, valinit=0, valstep=1)

fig4 = plt.figure(4)
varAx = plt.axes()
timeDelayVarChooser = RadioButtons(varAx, (0,1,2), active = 0)

def update(val):
    tao = taoChooser.val
    timeDelayVar = int(timeDelayVarChooser.value_selected)
    print(timeDelayVar)
    
    delay1 = states[:-tao*2,timeDelayVar]
    delay2 = states[tao:-tao,timeDelayVar]
    delay3 = states[tao*2:,timeDelayVar]

    ax1.clear()
    ax1.plot(delay1, delay2, delay3)
    fig2.canvas.draw()
    fig2.canvas.flush_events()

taoChooser.on_changed(update)
timeDelayVarChooser.on_clicked(update)

"""
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
"""
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")

plt.show()
