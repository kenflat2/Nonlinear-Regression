import numpy as np
import matplotlib.pyplot as plt

def normal(x, m, st):
    # return ( 1 / (st * np.sqrt(2 * np.pi))) * np.exp( ( -1 / 2 )*((x - m) / st) ** 2)
    return 2 ** ( -1 * (x - m) ** 2)
    
mean = 0
standardDev = 1

x = np.arange(-3,3,0.1)
y = np.zeros(60)
for i in range(60):
    y[i] = normal(x[i],mean,standardDev)

fig0 = plt.figure(0)
plt.plot(x,y)
plt.show()
