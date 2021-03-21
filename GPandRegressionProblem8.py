import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

# Covar Matrix and Mean function, define however you wish
def C(x1,x2):
    invlenscale = 10
    priorvar = 1
    # return priorvar * np.exp(-invlenscale * abs(x1 - x2)) # Exponential Covar
    return priorvar * np.exp(-(invlenscale ** 2) * (abs(x1 - x2) ** 2)) # Squared Exponential
    # return np.cos(8*np.pi*(x1-x2)) * np.exp(-invlenscale * abs(x1 - x2)) # Weird Covar
    # k = 1
    # return x * ( k - x ) * np.exp(-invlenscale / ( k - x2))#

def M(x):
    # return x*(1-x)
    return 0

# governing vars
min = 0
max = 1
step = 0.01
numDraws = 50

totalSteps = int(max * ( 1 / step))

r = np.arange(min, max, step)
mean = np.zeros((totalSteps,))
for i in range(totalSteps):
    mean[i] = M(r[i])

covm = np.zeros((totalSteps,totalSteps))
for i in range(totalSteps):
    for j in range(totalSteps):
        covm[i][j] = C(r[i],r[j])

R = 1
y = np.zeros((totalSteps,))
for i in range(totalSteps):
    y[i] = r[i]*np.exp(R-r[i])

fig1 = plt.figure(0)
for i in range(numDraws):
    GPsamp = rand.multivariate_normal(mean, covm)
    #print("GP Sample = ", GPsamp)
    plt.plot(GPsamp)

fig2 = plt.figure(1)
plt.imshow( covm, cmap = 'hot', interpolation='nearest')
plt.title("Covar Matrix as a heatmap")

fig3 = plt.figure(2)
plt.plot(r,y)

plt.show()
