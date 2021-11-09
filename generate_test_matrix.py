import numpy as np
import sys

M = int(sys.argv[1])
K = int(sys.argv[2])
N = int(sys.argv[3])
int8 = True if int(sys.argv[4]) == 1 else False
if int8:
    #scale = np.random.normal(size=(M)).astype(np.float32)
    scale = np.ones((M)).astype(np.float32) 
else:
    scale = 1
bias = np.random.normal(size=(M))
#bias = np.zeros((M))
AB = 1 + np.abs(np.random.normal(size=(K,M)).astype(np.float32) * 3)

BLOCK = int(sys.argv[5])
locs = [i for i in range(M* K) if i %BLOCK == 0]
zero_locs = np.random.choice(M*K//BLOCK, M * K // BLOCK // 10 * 9,replace=False) * BLOCK

for i in range(BLOCK):
    indices0 = np.unravel_index(zero_locs + i,(K,M))
    AB[indices0] = 0
#
AB = AB.transpose().copy()

#mask = (AB > 0) * 3
#AB = AB - mask
#print(AB)
#AB = AB * (AB > 2.7)
#BC = np.random.normal(size=(K,N)).astype(np.float32)
BC = np.ones((K,N)).astype(np.float32)
for i in range(K):
    BC[i] = np.random.randint(2)

if int8:
    AB = AB.astype(np.int8)
    BC = BC.astype(np.uint8)
    bias = bias.astype(np.int32)

AB = -AB


print("density",np.count_nonzero(AB) / M/ K)
AC = np.dot(AB,BC) + np.expand_dims(bias,1)
AC = AC.astype(np.float32) * np.expand_dims(scale,1)

if int8:
    #AC = AC.astype(np.int32)
    AC = AC.astype(np.int8)
if int8:
    np.save("bias.npy",bias)
else:
    np.save("bias.npy",bias.astype(np.float32))
np.save("matrix.npy",AB)
if int8:    
    np.save("scale.npy",scale)
np.save("matrix_transposed.npy",AB.transpose())
np.save("BC.npy",BC)
np.save("ref.npy",AC )
