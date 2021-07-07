import numpy as np
import sys
a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
print("Difference: ",np.sum(np.abs(a.squeeze()-b.squeeze())))
print(np.where(np.abs(a-b) > 0.01))
