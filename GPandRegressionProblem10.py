import numpy as np
import numpy.random as rand
import numpy.linalg as la
import matplotlib.pyplot as plt

# Covar Matrix and Mean function, define however you wish
def C(x1,x2):
    invlenscale = 3
    priorvar = 10
    # return priorvar * np.exp(-invlenscale * abs(x1 - x2)) # Exponential Covar
    return priorvar * np.exp(-(invlenscale ** 2) * (abs(x1 - x2) ** 2)) # Squared Exponential
    # return np.cos(8*np.pi*(x1-x2)) * np.exp(-invlenscale * abs(x1 - x2)) # Weird Covar
    # k = 1
    # return x * ( k - x ) * np.exp(-invlenscale / ( k - x2))#

def M(x):
    # return x*(1-x)
    return 0

# Create GP
min = -5
max = 5
step = 0.1
numDraws = 25

totalSteps = int((max - min) * ( 1 / step))

X = np.arange(min, max, step)
mean = np.zeros((totalSteps,))
for i in range(totalSteps):
    mean[i] = M(X[i])


# Create Data
mind = -5
maxd = 5
stepd = 0.25
totalStepsd = int((maxd - mind) * ( 1 / stepd ))

stdev = 0.5

xd = np.arange(mind, maxd, stepd)
yd = np.zeros((totalStepsd,))
for i in range(totalStepsd):
    newYd = (xd[i] ** 3) * np.exp(-(xd[i] ** 2)) + xd[i] ** 2 - xd[i] -0.5 * xd[i] ** 6
    #newYd = 2 * xd[i] ** 3 + 4 * xd[i] ** 2 - 3 * xd[i] +6
    yd[i] = newYd
yd = np.nan_to_num(yd,nan=0,posinf=100,neginf=100)

md = np.zeros((totalStepsd,))
for i in range(totalStepsd):
    md[i] = M(xd[i])
    
C3 = yd - md

K = np.zeros((totalStepsd,totalStepsd))
for i in range(totalStepsd):
    for j in range(totalStepsd):
        K[i][j] = C(xd[i],xd[j])

C2 = np.zeros((totalStepsd,totalSteps))
for i in range(totalStepsd):
    for j in range(totalSteps):
        C2[i][j] = C(xd[i],X[j])

C1 = np.zeros((totalSteps,totalSteps))
for i in range(totalSteps):
    for j in range(totalSteps):
        C1[i][j] = C(X[i],X[j])

print("C1 ", C1.shape, " C2 ", C2.shape, " C3 ", C3.shape, " m ", mean.shape, " K ", K.shape)
print("C1 ", C1, " C2 ", C2, " C3 ", C3, " m ", mean, " K ", K)
print(" inverse ",la.inv(K + np.identity(totalStepsd) * (stdev ** 2)))
# Ef = mean + np.transpose(C2) @ la.inv(K + np.identity(totalStepsd) * (stdev ** 2)) @ C3
Ef = np.transpose(C2) @ la.inv(K + np.identity(totalStepsd) * (stdev ** 2)) @ C3
Ev = C1 - np.transpose(C2) @ la.inv(K + np.identity(totalStepsd) * (stdev ** 2)) @ C2

v = np.zeros((totalSteps,))
for i in range(totalSteps):
    v[i] = Ev[i][i]
print("Ef = ", Ef)
print("Ev = ",Ev)

fig1 = plt.figure(0)
for i in range(numDraws):
    GPsamp = rand.multivariate_normal(mean, Ev)
    #print("GP Sample = ", GPsamp)
    plt.plot(GPsamp)

fig2 = plt.figure(1)
plt.imshow( Ev, cmap = 'hot', interpolation='nearest')
plt.title("Covar Matrix as a heatmap")

fig3 = plt.figure(2)
plt.plot(xd,yd)
plt.plot(X,Ef)
plt.plot(X,Ef-v)
plt.plot(X,Ef+v)
# plt.axis([-5,5,-10,10])
plt.title("Input Data")

plt.show()
