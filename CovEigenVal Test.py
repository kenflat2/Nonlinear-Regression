import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

expCov = np.array([[1, 0.1, 0.9],[0.1, 1, 0.2],[0.9, 0.2, 1]])
print(expCov)
expVal, expVec = la.eigh(expCov)
print("Eigenvalues ", expVal)
print("Eigenvectors \n", expVec)

scaledVec = expVec * expVal
print(scaledVec, scaledVec[:,0])

b = 1
covAx = plt.figure().add_subplot(projection="3d")
covAx.axes.set_xlim3d(left=-b, right=b)
covAx.axes.set_ylim3d(bottom=-b, top=b)
covAx.axes.set_zlim3d(bottom=-b, top=b)
covAx.quiver(np.zeros((3)),np.zeros((3)),np.zeros((3)), scaledVec[0,:], scaledVec[1,:], scaledVec[2,:], normalize=False)
plt.show()
