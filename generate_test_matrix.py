import numpy as np
import sys

M = int(sys.argv[1])
K = int(sys.argv[2])
N = int(sys.argv[3])
int8 = True if int(sys.argv[4]) == 1 else False

bias = np.random.normal(size=(M))
AB = np.random.normal(size=(M,K)).astype(np.float32)
AB = AB * (AB > 0.25)
#BC = np.random.normal(size=(K,N)).astype(np.float32)
BC = np.ones((K,N)).astype(np.float32)
for i in range(K):
    BC[i] = np.random.randint(2)

if int8:
    AB = AB.astype(np.int8)
    BC = BC.astype(np.uint8)
    bias = bias.astype(np.int32)

AC = np.dot(AB,BC) + np.expand_dims(bias,1)
if int8:
    #AC = AC.astype(np.int32)
    AC = AC.astype(np.int8)

np.save("bias.npy",bias.astype(np.float32))
np.save("matrix.npy",AB)
np.save("matrix_transposed.npy",AB.transpose())
np.save("BC.npy",BC)
np.save("ref.npy",AC)
