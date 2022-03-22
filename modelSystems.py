import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.integrate import odeint

def linearSystem(x,t):
    A = np.array([[1/3,-1/2],[5/4,3/2]])
    
    return A @ x

def Mease(x,t):
    a = 0.01
    
    (x1,x2) = x
    
    x1p = -x1-a*(x1+19*x2)*(x1+x2)-18*a**2*(x1+x2)**3
    x2p = -10*x2-2*a*(4*x1-5*x2)*(x1+x2)+18*a**2*(x1+x2)**3
    
    return (x1p, x2p)

def Logistic(x, t):
    r = 4
    return r * x * (1-x)

def LogisticP(x, t, r):
    return r(t) * x * (1-x)

def DensityDependentMaturation(x, t):
    s = 0.02 # 
    gamma = 0.01 # rate at which growth is impeded by density
    sA = 0.1 # survival of adults
    sJ = 0.5 # survival of children
    b = 40 # juveniles produced by adult
    Gmax = 0.9 # maximum growth rate
    g = lambda x : Gmax*np.exp(-gamma*x) # juvenile to adult rate(growth?)

    At = x[0] # num of adults
    Jt = x[1] # num of juveniles

    # zt = -0.0001 # rand.normal(-s/2, s)

    # m = np.array([[ sA,             sJ*g(At+Jt)     ],
    #              [ b*np.exp(zt),   sJ*(1-g(At+Jt)) ]])

    m = np.array([[ sA,             sJ*g(At+Jt)     ],
                  [ b,   sJ*(1-g(At+Jt)) ]])

    return m @ x.T
    
def DensityDependentMaturationP(x, t, Gmax):
    s = 0.02
    gamma = 0.01
    sA = 0.1
    sJ = 0.5
    b = 35
    
    g = lambda x , t: Gmax(t)*np.exp(-gamma*x)

    At = x[0]
    Jt = x[1]

    zt = -0.02 # rand.normal(-s/2, s)

    m = np.array([[ sA,             sJ*g(At+Jt,t)       ],
                  [ b*np.exp(zt),   sJ*(1-g(At+Jt,t))]  ])

    return m @ x.T

## Models ##
def RosenzweigMacArthurP(x, t, h2):
    c1 = 5.0
    h1 = 3.0
    c2 = 0.1
    m2 = 0.4
    m3 = 0.008
    
    dx = x[0]*(1-x[0]) - c1 * x[0] * x[1] / (1.0 + h1*x[0])
    dy = c1 * x[0] * x[1] / (1.0 + h1 * x[0]) - c2 * x[1] * x[2] / (1.0 + h2(t) * x[1]) - m2*x[1]
    dz = c2 * x[1] * x[2] / (1.0 + h2(t) * x[1]) - m3 * x[2]

    return (dx, dy, dz)

def RosenzweigMacArthur(x, t):
    c1 = 5.0
    h1 = 3.0
    c2 = 0.1
    m2 = 0.4
    m3 = 0.008
    h2 = 2.0
    
    dx = x[0]*(1-x[0]) - c1 * x[0] * x[1] / (1.0 + h1*x[0])
    dy = c1 * x[0] * x[1] / (1.0 + h1 * x[0]) - c2 * x[1] * x[2] / (1.0 + h2 * x[1]) - m2*x[1]
    dz = c2 * x[1] * x[2] / (1.0 + h2 * x[1]) - m3 * x[2]

    return (dx, dy, dz)

def Lorenz96P(x, t, F):
    N = 5 # dimension

    # Setting up vector
    d = np.zeros(N)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F(t)
    return d

def LorenzP(xi, t, rho, sigma, beta):
    
    (x,y,z) = xi
    return sigma(t) * (y - x), x * (rho(t) - z) - y, x * y - beta(t) * z  # Derivatives

def Lorenz(xi,t):
    rho = 25.0
    sigma = 10.0
    beta = 8.0 / 3.0
    
    (x,y,z) = xi
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def RosslerP(xi, t, a, b, c):    
    (x,y,z) = xi

    dx = -y - z
    dy = x + a(t) * y
    dz = b(t) + z * ( x - c(t) )

    return np.array( [dx,dy,dz] )

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

def HastingsPowell(xi,t):
    (x,y,z)=xi


    a1 = 5
    a2 = 0.1
    b1 = 3
    b2 = 2
    d1 = 0.4
    d2 = 0.01

    dx = x*(1-x)- a1*x*y/(1+b1*x)
    dy = a1*x*y/(1 + b1*x) - d1*y - a2*y*z/(1 + b2*y)
    dz = a2*y*z/(1 + b2*y) - d2*z

    return dx, dy, dz

def HastingsPowellP(xi, t, ):
    (x,y,z)=xi


    a1 = 5
    a2 = 0.1
    b1 = 3
    b2 = 2
    d1 = 0.4
    d2 = 0.01

    dx = x*(1-x)- a1*x*y/(1+b1*x)
    dy = a1*x*y/(1 + b1*x) - d1*y - a2*y*z/(1 + b2*y)
    dz = a2*y*z/(1 + b2*y) - d2*z

    return dx, dy, dz
    

"""
def test(f):
    print(f)
    return f
"""

# just one function that should take care of all my integrating needs
def generateTimeSeriesContinuous(f, t0, tlen=256, end=32, reduction=1, settlingTime=0, nsargs=None):
    t = np.linspace(0,end,num=tlen+settlingTime)
    
    F = globals()[f]

    if nsargs == None:
        ts = odeint(F, t0, t)[settlingTime::reduction]
    else:
        ts = odeint(F, t0, t, args=nsargs)[settlingTime::reduction]

    return ts

def generateTimeSeriesDiscrete(f, t0, tlen=256, settlingTime=0, nsargs=None):
    if type(t0) == float:
        ts = np.zeros((tlen,1))
    else:
        ts = np.zeros((tlen,t0.shape[0]))

    ts[0] = t0

    # allow system to settle
    for i in range(settlingTime):
        if nsargs==None:
            ts[0] = f(ts[0], 0)
        else:
            ts[0] = f(ts[0], 0, *nsargs)

    # now evaluate
    if nsargs==None:
        for i in range(1,tlen):
            ts[i] = f(ts[i-1], i)
    else:
        for i in range(1,tlen):
            ts[i] = f(ts[i-1], i, *nsargs)

    return ts

