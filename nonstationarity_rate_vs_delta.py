import numpy as np
import numpy.linalg as la
import numpy.random as rand
from scipy.integrate import odeint
from scipy import stats
from EDMtoolkit import *
from modelSystems import *
import datetime

settlingTime = 2 ** 12
tlen = 2 ** 8
end = 2**11 # (3.498901098901099 / (12*reduction)) * tlen # 2**3
reduction = 2 ** 2

t = np.linspace(0, end, num=tlen)
ts = t / end

resolution = 10
horizon = 1
num_repetitions = 10
noise_magnitude = 0.0
maxLags = 10

# [rate, lamdbaLR, stationary probability]
# column 0 is nonstationarity rate,
# columns 1-10 are deltas for each E
# columns 11-20 are likelihoods for NSMap
# columns 21-30 are likelihoods for SMap

table = np.zeros((resolution*num_repetitions, 31))

x0 = np.array([0.1,0.4,9]) 
for r in range(resolution):
    rate = float(r)/resolution
    b1 = lambda t: 2.5 + rate * 4 * t / end
    
    for i in range(num_repetitions):
        x0 = np.array([0.1,0.4,9]) + np.multiply(np.array([1,0.5,7]),rand.random(3))
        Xr = standardize(generateTimeSeriesContinuous('HastingsPowellP', x0, nsargs=(b1,), end=end, tlen = tlen, reduction = reduction, settlingTime=settlingTime))[:,0,None]
            
        ## Add Noise ##
        # Xr += rand.random((Xr.shape[0],1)) * np.ptp(Xr) * noise_magnitude

        T2 = StationaryProbability(Xr, ts, horizon, maxLags)
        
        table[r*num_repetitions+i] = np.hstack([np.array([rate]),T2[:,4], T2[:,0], T2[:,1]])


datetime_str = datetime.datetime.now().strftime("%m-%d-%Y:%X") 
np.save(f"nonstationarity_vs_delta{datetime_str}", table)