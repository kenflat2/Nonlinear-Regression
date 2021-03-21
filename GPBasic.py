from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

# These are our constants
N = 5  # Number of variables
F = 7  # Forcing


def L96(x, t):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def C(X1,X2):
    invlenscale = 0.1
    tao = 2
    return tao * np.exp(-(invlenscale ** 2) * np.dot(X1-X2,X1-X2))

def GPPrediction(x, xd, yd, K):
    priorVar = 0.1
    
    insize = np.shape(xd)[0]

    C1 = np.zeros(insize)
    for i in range(insize):
        C1[i] = C(x,xd[i])

    # print(np.shape(C1)," ", np.shape(la.inv(K + np.identity(insize)))," ", np.shape(yd))
    pred = np.transpose(C1) @ la.inv(K + np.identity(insize) * priorVar) @ yd
    # print(" yd ", yd, " C1 ", C1)
    return pred

x0 = F * np.ones(N)  # Initial state (equilibrium)
x0[0] += 0.01  # Add small perturbation to the first variable
t = np.arange(0.0, 10.0, 0.05)

x = odeint(L96, x0, t)
md = np.mean(x,axis=0)
stdd = np.std(x,axis=0)
for i in range(np.shape(x)[0]):
    x[i] = (x[i] - md) / stdd
x0 = (x0 - md) / stdd
print("mean ",md," stdd ", stdd)

X = x[:-1,]
Y = x[1:,]

# initState = np.array([0,0,0,1,0])

insize = np.shape(X)[0]
K = np.zeros((insize,insize))
for i in range(insize):
    for j in range(insize):
        K[i][j] = C(X[i],X[j])


pred = x0
print("prediction ",pred)

#PREDICTION ITERATION(my favorite)
iterPred = [pred]
for i in range(50):
    pred = GPPrediction(pred,X,Y,K)
    iterPred.append(pred)
iterPred = np.array(iterPred)
print("iterPred ",iterPred)

print("X ",X.shape," Y ", Y.shape)

# 1 STEP PREDICTION ACCURACY
errList = []
onesteplist = [x0]
p=x0
for i in range(insize-1):
    p = GPPrediction(p,X,Y,K)
    onesteplist.append(p)
    diff = X[i+1] - p
    err = np.dot(diff, diff)
    errList.append(err)
onesteplist = np.array(onesteplist)

# Plot the first three variables
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 4])
ax.plot(iterPred[:, 0], iterPred[:, 1], iterPred[:, 2])
ax.plot(onesteplist[:, 0], onesteplist[:, 1], onesteplist[:, 2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")

fig0 = plt.figure(0)
ax0 = fig0.add_subplot()
ax0.plot(range(len(errList)),errList)

plt.show()
