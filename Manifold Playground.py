import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons


# Rossler System
def Rossler(xi, t):
    a = 0.2
    b = 0.2
    c = 5.7
    
    (x,y,z) = xi

    dx = -y - z
    dy = x + a * y
    dz = b + z * ( x - c )

    return np.array( [dx,dy,dz] )

def RosslerP(xi, t, a, b, c):    
    (x,y,z) = xi

    dx = -y - z
    dy = x + a * y
    dz = b + z * ( x - c )

    return np.array( [dx,dy,dz] )

def Sprott(xi, t):
    (x,y,z) = xi
    return ( y,-x - np.sign(z)*y, y**2 - np.exp(-x**2))

"""
def SprottP(xi, t, d):
    (x,y,z) = xi
    return ( y, -x - np.sign(z)*y, y**2 - d*np.exp(-x**2))
"""

def SprottP(xi, t, d):
    (x,y,z) = xi
    return ( y/d, -x - np.sign(z)*y*d, y**(2) - d*np.exp(-x**2))

def Lorenz(xi,t):
    rho = 25.0
    sigma = 10.0
    beta = 8.0 / 3.0
    
    (x,y,z) = xi
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def LorenzP(xi,t, rho, sigma, beta):
    
    (x,y,z) = xi
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def Lorenz96(x, t):
    N = 5 # dimension
    F = 8

    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d    

def Lorenz96P(x, t, F):
    N = 5 # dimension

    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d   

def RosenzweigMacArthur(x, t, p):    
    dx = x[0]*(1-x[0]) - p["c1"] * x[0] * x[1] / (1.0 + p["h1"]*x[0])
    dy = p["c1"] * x[0] * x[1] / (1.0 + p["h1"] * x[0]) - p["c2"] * x[1] * x[2] / (1.0 + p["h2"] * x[1]) - p["m2"]*x[1]
    dz = p["c2"] * x[1] * x[2] / (1.0 + p["h2"] * x[1]) - p["m3"] * x[2]

    return (dx, dy, dz)

end = 2 ** 7
tlen = 2 ** 13
trainToTest = 0.8 # between 0 and 1
t = np.linspace(0, end, num = tlen)

# MAKE SURE TO UPDATE THE DIMENSION WHEN SWITCHING ATTRACTORS
dim = 3
t0 = np.array([1,5,15])
# t0 = np.array([1,1,1])# np.zeros(dim) * 2
# t0 = np.array([0.8,0.1,9])
# t0 = np.ones(dim)
t0[0] += 0.1

# include the number of parameters
nParams = 0
embst = 1
"""
params = {"c1":5.0,
     "h1":3.0,
     "c2":0.1,
     "h2":2.0,
     "m2":0.4,
     "m3":0.008}

params = {"F" : 8}

params = {"d" : 1}
"""

params = {"rho" : 28.0,
          "sigma" : 10,
          "beta" : 8/3}

p0 = list(params.keys())[0]

print(t0)

# STATIONARY SIMULATION VERSION: UPDATE ATTRACTOR YOU WANT HERE
#               \/\/\/\/
# states = odeint(RosenzweigMacArthur,t0,t,args=(params,))
# states = odeint(Sprott,t0,t)
# states = odeint(LorenzP,t0,t, args=(0.5,))

def plotData(X):
    ax2.clear()
    ax2.plot(X[:,0],X[:,1],X[:,2], alpha = 1, c="black")
    # ax2.plot(s[:-2*embst,0], s[1*embst:-1*embst,0], s[2*embst:,0], alpha=1, c="black")
    ax2._axis3don = True
    ax2.set_facecolor("white")

def makeData():
    return odeint(LorenzP, t0, t, args=tuple(params.values()))
    # return odeint(Lorenz96P,t0,t, args=tuple(params.values()))
states = makeData()

# END STATIONARY SIMULATIONS

# FROM DATA
"""
file = "GPDD.csv"
data = pd.read_csv(file,encoding="utf-8",na_filter=False)
states = data.to_numpy()
print(states)
"""
# END FROM DATA


# NON STATIONARY VERSIONS

"""
# Sprott
d = lambda t : np.exp(-t)

largs = lambda t : tuple(d(t))

states = np.zeros((tlen,3))
states[0] = t0
for i in range(1, tlen ):
    # print(largs(i))
    states[i] = odeint(SprottP,states[i-1],np.array([t[i-1],t[i]]),args=largs(i))[1,:]
Xr = standardize(states[settlingTime:])


p1 = 0.2
p2 = 0.2
p3 = 5.7

deltaP = 0.01
states = np.zeros((tlen,3))
states[0] = t0
for i in range(1, tlen ):
    states[i] = odeint(LorenzP,states[i-1],np.array([0,step]),args=(p1,p2,p3))[1,:]
    p2 += deltaP
"""
# END NON STATIONARY

# Normalize, split data
states = (states - states.mean(0)) / states.std(0) # normalize
testTrainSplit = int(states.shape[0] * trainToTest)

X = states
# X = np.log(states[:,2,None]+1)
# Y = np.log(states[1:,])

# Print Input
fig2 = plt.figure(2)
ax2 = fig2.gca(projection="3d")
ax2._axis3don = True
ax2.set_facecolor("white")

if dim == 2:
    ax2 = plt.subplot()
    ax2.plot(X[:,0],X[:,1])
else:
    plotData(states)
    #ax2.plot(X[:-2*embst,0],X[1*embst:-1*embst,0],X[2*embst:,0], alpha = 1, c="black")
    
# ax2.plot(X[:,0], X[:,1], X[:,2])
# ax2.plot(X[:-2*embst,0],X[1*embst:-1*embst,0],X[2*embst:,0], alpha = 1, c="black")


# user interaction stuff
fig3 = plt.figure(3)
sliderAx = plt.axes()
slider = Slider(sliderAx, "Slider", 0, 5, valinit=list(params.values())[0], valstep=0.1)

fig4 = plt.figure(4)
paramAx = plt.axes()
paramButtons = RadioButtons(paramAx, params.keys(), active = 0)

def update(val):
    global p0, params
    # global p1, p2, p3
    params[p0] = slider.val
        # p1 = slider.val
    # elif (paramButtons.value_selected == "b"):
    #     p2 = slider.val
    # elif (paramButtons.value_selected == "c"):
    #     p3 = slider.val

    # Line Plot Update
    # s = odeint(Sprott,t0,t)
    # s = odeint(SprottP,t0,t, args=(p1,))

    s = makeData()
    plotData(s)
    # ax2.plot(s[:,0], s[:,1], s[:,2], alpha=0.5)

    # Quiver Update
    mi = np.nanmin(s,axis=0)
    ma = np.nanmax(s,axis=0)
    st = 10

    # x, y, z = np.meshgrid(np.arange(mi[0],ma[0],st[0]), np.arange(mi[1],ma[1],st[1]),np.arange(mi[2],ma[2],st[2]))
    # u, v, w = (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    # ax2.quiver(x,y,z,u,v,w,length=5,normalize=True, color = "r", alpha = 0.25)

    # x, y, z = np.meshgrid(np.linspace(mi[0],ma[0],num=st), np.linspace(mi[1],ma[1],num=st),np.linspace(mi[2],ma[2],num=st))
    # u, v, w = SprottP((x,y,z),0,p1)
    # ax2.quiver(x,y,z,u,v,w,length=0.5,normalize=True, color = "r", alpha=0.25)
    
    fig2.canvas.draw()
    fig2.canvas.flush_events()

def update2(val):
    global p0
    p0 = val
    slider.set_val(params[p0])
    fig3.canvas.draw()
    fig3.canvas.flush_events()
    
    # if (paramButtons.value_selected == "d"):
    #     slider.set_val(p1)
    # elif (paramButtons.value_selected == "b"):
    #     slider.set_val(p2)
    # elif (paramButtons.value_selected == "c"):
    #     slider.set_val(p3)

slider.on_changed(update)
paramButtons.on_clicked(update2)

# Quiver Plot
fig2 = plt.figure(2)
ax2 = fig2.gca(projection="3d")

mi = np.nanmin(X,axis=0)
ma = np.nanmax(X,axis=0)
st = 10
print(st)

# x, y, z = np.meshgrid(np.arange(mi[0],ma[0],st[0]), np.arange(mi[1],ma[1],st[1]),np.arange(mi[2],ma[2],st[2]))
# u, v, w = (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
# ax2.quiver(x,y,z,u,v,w,length=st[0],normalize=True, color = "r", alpha=0.25)

# x, y, z = np.meshgrid(np.linspace(mi[0],ma[0],num=st), np.linspace(mi[1],ma[1],num=st),np.linspace(mi[2],ma[2],num=st))
# u, v, w = Sprott((x,y,z),0)
# ax2.quiver(x,y,z,u,v,w,length=0.5,normalize=True, color = "r", alpha=0.25)

plt.show()
