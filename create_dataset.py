import numpy as np
import scipy as sp
import random
import cmath
import math

# Define the size of our data vector as N, as well as the sparsity of the time-domain representation of our vector x
N = 100
k = 10

# Then build the vector x in the time-domain, we generate k random indices from 0-99, and fill our x vector with random values from 0-10

amp = 10
indices = random.sample(range(0,99),k)
amps = random.sample(range(0,amp),k)
x = np.zeros((N,))

for indx in range(k):
    x[indices[indx]] = amps[indx]

# Then define the (unitary, note the scaling) DFT matrix

F = sp.linalg.dft(N)/np.sqrt(N)
Fstar = F.conj().T
identity = F@Fstar

# Notice that due to numerical errors, the resulting matrix is not unitary
# Then generate h by taking the DFT of x, notice how h is now dense!

h = F@x

# This was done once, now we need to generate multiple thousands of h to train the dataset. 
# 

# Now train the autoencoder