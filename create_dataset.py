import numpy as np
import scipy as sp
import random
import cmath
import math

# Define the size of our data vector as N, as well as the sparsity of the time-domain representation of our vector x
N = 100
k = 10

# Then build the vector x in the time-domain, we generate k random indices from 0-99, and fill our x vector with random values from 0-10

def buildDataSet(max_amplitude,min_sparsity,max_sparsity,vector_size,data_set_size):
    sparse_data = np.zeros((vector_size,data_set_size)) # Initialize the sparse_data matrix

    # Iterate over the columns of the sparse_data matrix to define the data samples
    for i in range(data_set_size):
        sparsity = random.randint(min_sparsity,max_sparsity)
        indices = random.sample(range(vector_size),sparsity)
        amps = random.sample(range(max_amplitude),sparsity)
        sparse_data[indices,i] = amps
    
    # Define the DFT matrix and multiply our spare_data vectors with it to find dense data
    DFT = sp.linalg.dft(vector_size)/np.sqrt(vector_size)
    dense_data = DFT@sparse_data
    return dense_data,sparse_data

max_amplitude = 10
min_sparsity = 3
max_sparsity = 5
vector_size = 100
data_set_size = 1000
dense_data, sparse_data = buildDataSet(max_amplitude,min_sparsity,max_sparsity,vector_size,data_set_size)    

# This was done once, now we need to generate multiple thousands of h to train the dataset. 
# Some questions would be:
# How large do we make the dataset
# Does the generated data need to be sparse at the same indices? (I think so, otherwise the trained decoder would not function properly)

# Now train the autoencoder

DFT = sp.linalg.dft(vector_size)/np.sqrt(vector_size)