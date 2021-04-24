import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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

begin = 0
end = 10
step = 0.05
tlen = int((end-begin)/step)
trainToTest = 0.8 # between 0 and 1
t = np.arange(begin, end, step)

# MAKE SURE TO UPDATE THE DIMENSION WHEN SWITCHING ATTRACTORS
dim = 3
t0 = np.array([1,1,7])# np.zeros(dim) * 2
t0[0] += 0.1

# STATIONARY SIMULATION VERSION: UPDATE ATTRACTOR YOU WANT HERE
#               \/\/\/\/
states = odeint(Rossler,t0,t)
# END STATIONARY SIMULATION

# FROM DATA
"""
file = "paramecium_didinium - cleaned.csv"
data = pd.read_csv(file,encoding="utf-8",na_filter=False)
states = data.to_numpy()
print(states)
"""
# END FROM DATA


# NON STATIONARY VERSION

p1 = 0.2
p2 = 0.2
p3 = 5.7

"""
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

X = states[:-1,]
Y = states[1:,]

# Print Input
fig2 = plt.figure(2)
ax2 = fig2.gca(projection="3d")
if dim == 2:
    ax2 = plt.subplot()
    ax2.plot(X[:,0],X[:,1])
else:
    ax2.plot(X[:,0],X[:,1],X[:,2])

# user interaction stuff
fig3 = plt.figure(3)
sliderAx = plt.axes()
slider = Slider(sliderAx,"Tao", 0, 8, valinit=0, valstep=0.1)

fig4 = plt.figure(4)
paramAx = plt.axes()
paramButtons = RadioButtons(paramAx, ("a","b","c"), active = 0)

def update(val):
    global p1, p2, p3
    if (paramButtons.value_selected == "a"):
        p1 = slider.val
    elif (paramButtons.value_selected == "b"):
        p2 = slider.val
    elif (paramButtons.value_selected == "c"):
        p3 = slider.val

    # Line Plot Update
    s = odeint(RosslerP,t0,t, args=(p1,p2,p3))
    ax2.clear()
    ax2.plot(s[:,0], s[:,1], s[:,2])

    # Quiver Update
    mi = np.nanmin(s,axis=0)
    ma = np.nanmax(s,axis=0)
    st = abs(mi - ma) / 5
    print(st)

    # x, y, z = np.meshgrid(np.arange(mi[0],ma[0],st[0]), np.arange(mi[1],ma[1],st[1]),np.arange(mi[2],ma[2],st[2]))
    # u, v, w = (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    # ax2.quiver(x,y,z,u,v,w,length=5,normalize=True, color = "r", alpha = 0.25)
    
    fig2.canvas.draw()
    fig2.canvas.flush_events()

def update2(val):
    global p1, p2, p3
    if (paramButtons.value_selected == "a"):
        slider.set_val(p1)
    elif (paramButtons.value_selected == "b"):
        slider.set_val(p2)
    elif (paramButtons.value_selected == "c"):
        slider.set_val(p3)

slider.on_changed(update)
paramButtons.on_clicked(update2)

# Quiver Plot
fig2 = plt.figure(2)
ax2 = fig2.gca(projection="3d")


mi = np.nanmin(X,axis=0)
ma = np.nanmax(X,axis=0)
st = abs(mi - ma) / 5
print(st)

# x, y, z = np.meshgrid(np.arange(mi[0],ma[0],st[0]), np.arange(mi[1],ma[1],st[1]),np.arange(mi[2],ma[2],st[2]))
# u, v, w = (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
# ax2.quiver(x,y,z,u,v,w,length=st[0],normalize=True, color = "r", alpha=0.25)

plt.show()
