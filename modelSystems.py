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

### MULTIDIMENSIONAL TENT MAP ###
# proof of correctness: we wish for the first map to act on its own,
# the following uses the current value of the previous as its midpoint,
# so on an so on until some end. The length of the input vector 
# determines the number of dimensions.
def Tents(x, t):
    x_new = np.zeros(x.shape)

    x_new[0] = Tent(x[0], t, 0.5)

    # the first one is an unmodified tent map, the latter uses the value
    # of the former for its midpoint.
    for i in range(1, x.shape[0]):
        x_new[i] = Tent(x[i], t, x[i-1])

    return x_new

# a single tent map where the midpoint is specified
def Tent(x, t, M):

    if type(M) == np.dtype("f8"):
        assert M >= 0 and M <= 1
        m = M
    else:
        m = M(t)

    # Proof of Correctness: the only thing we need to check is that we
    # never divide by 0. If m is neither 0 nor 1 then this will never
    # happen. If m is 0 then if x is > 0 we just return 1-x, and if
    # x is 0 then x==m and we correctly return 1. If m is 1 and x<1
    # then the second condition is chosen and we are fine, if x==1 then
    # we again return 1. //

    if x == m:
        return 1
    elif x < m:
        return x / m
    elif x > m:
        return (1-x) / (1-m)


def Logistic(x, t):
    r = 4
    return r * x * (1-x)

def LogisticP(x, t, r):
    return r(t) * x * (1-x)

def LogisticIslandsP(x, t, r, m):
    I = x.shape[0]
    xr = x.copy()
    
    for i in range(I):
        ip = (i + 1) % I
        im = (i - 1) % I

        xr[i] = LogisticP(x[i]*(1-m(t))+(x[ip]+x[im])*m(t)/2, t, r)

    return xr

def LogisticIslands(x, t):
    r = lambda t : 4
    m = lambda t : 0.5
    
    return LogisticIslandsP(x, t, r, m)

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

def HastingsPowellP(xi, t, b1):
    (x,y,z)=xi


    a1 = 5
    a2 = 0.1
    b1 = b1(t)
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
    F = globals()[f]

    if settlingTime > 0:
        tSettle = np.arange(0,settlingTime, step=end/(reduction*tlen))

        # let the system settle
        if nsargs == None:
            x0 = odeint(F, t0, tSettle)[-1]
        else:
            driver_settle = tuple([lambda t: x(0) for x in nsargs])
            x0 = odeint(F, t0, tSettle, args=driver_settle)[-1]
    else:
        x0 = t0
    
    t = np.linspace(0,end,num=tlen*reduction)
    
    if nsargs == None:
        ts = odeint(F, x0, t)[::reduction]
    else:
        ts = odeint(F, x0, t, args=nsargs)[::reduction]

    return ts

def generateTimeSeriesDiscrete(f, t0, tlen=256, settlingTime=0, nsargs=None):
    F = globals()[f]
    
    if type(t0) == float:
        ts = np.zeros((tlen,1))
    else:
        ts = np.zeros((tlen,t0.shape[0]))

    ts[0] = t0

    # allow system to settle
    for i in range(settlingTime):
        if nsargs==None:
            ts[0] = F(ts[0], 0)
        else:
            ts[0] = F(ts[0], 0, *nsargs)

    # now evaluate
    if nsargs==None:
        for i in range(1,tlen):
            ts[i] = F(ts[i-1], i)
    else:
        for i in range(1,tlen):
            ts[i] = F(ts[i-1], i, *nsargs)

    return ts

