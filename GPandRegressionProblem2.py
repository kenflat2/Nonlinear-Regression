import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

n = 50
a = 0.2

z1 = np.random.normal(loc=0,scale=4/3,size=1) # Munch's Randomness
# z1 = np.random.normal(loc=0,scale=1/30,size=1)
THESYSTEM = [z1]

for i in range(0,n-1):
    zn = THESYSTEM[-1]
    n1 = np.random.normal(loc=0,scale=1,size=1)
    znp1 = a * zn + n1
    THESYSTEM.append(znp1)

x = np.random.uniform(low=-2,high=2,size=50)
print(x, " Shape ", np.shape(x))
y = np.array(list(map(lambda z, x: z + x + 1, THESYSTEM, x)))
y = np.transpose(y)[0]
print(y," Shape ", np.shape(y))

""" Linear Solution """
xavg = np.average(x) # shitty covar calculation
yavg = np.average(y)
cov = np.array([],dtype=np.float64)
for (xi,yi) in zip(x,y):
    cov = np.append(cov,[(xi-xavg)*(yi-yavg)])
cov = np.average(cov)
cov2 = np.cov(np.vstack((y,x)))

var = np.var(x) # determine a and b
b = cov / var
A = np.average(y)-b*np.average(x)
print("b = ",b)
print("a = ", A)

""" General Least Squares """
# X = np.vstack((x,y))
# Y = np.delete(X,(0),axis=1)
# X = np.delete(X, (-1), axis=1)
# sigma = np.cov(X) # use numpy's covar matrix before Steve's
x = np.reshape(x,(1,50))
y = np.reshape(y,(1,50))

sigma = np.fromfunction(lambda i, j: (pow(a,abs(i-j)))/(1-pow(a,2)), (50,50), dtype=np.float64)
print("x trans shape ", np.shape(np.transpose(x)), ' sigma shape ', np.shape(sigma), " x shape ", np.shape(x))

Bhat = la.inv(x @ la.inv(sigma)  @ np.transpose(x) ) @ (x @ la.inv(sigma) @ np.transpose(y))
print("Bhat shape ", np.shape(Bhat))
print("Bhat ",Bhat)
Ahat = np.average(y) - Bhat*np.average(x)

# Graph it!
x = np.transpose(x)

fig1 = plt.figure(1)
plt.plot(THESYSTEM)
plt.title("Nonlinear Dynamics")

fig2 = plt.figure(2)
plt.plot(x)
plt.title("Truly Random")

fig3 = plt.figure(3)
plt.scatter(x,y)
plt.plot(x, A + b*x,"-")
plt.plot(x, Ahat + Bhat*x,"-")
plt.legend(["Naive Regression","General Regression"])
plt.title("Y")

plt.show()
