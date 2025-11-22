import random

import numpy as np
import scipy as sp
import torch
from torch.utils.data import TensorDataset, DataLoader

def buildDataSet(max_amplitude, min_sparsity, max_sparsity, vector_size, data_set_size):
    sparse_data = np.zeros((vector_size, data_set_size), dtype=float)  # Ensure float type

    # Iterate over the columns of the sparse_data matrix to define the data samples
    for i in range(data_set_size):
        sparsity = random.randint(min_sparsity, max_sparsity)
        indices = random.sample(range(vector_size), sparsity)
        amps = np.random.uniform(-max_amplitude, max_amplitude, sparsity)  # Use negative and positive values
        sparse_data[indices, i] = amps

    # Define the DFT matrix and multiply our sparse_data vectors with it to find dense data
    DFT = sp.linalg.dft(vector_size) / np.sqrt(vector_size)
    dense_data = DFT @ sparse_data

    return dense_data, sparse_data

def Generate_Dataloader(max_amplitude,min_sparsity,max_sparsity,vector_size,data_set_size):
    # Function takes as inputs the (maximum) amplitude of the signal, minimum and maximum sparsity of the signal, the signal length (vector_size) and the size of the dataset
    # Outputs dataloader and signal variance, which is important for normalization
    dense_data, sparse_data = buildDataSet(max_amplitude,min_sparsity,max_sparsity,vector_size,data_set_size)

    X = np.concatenate((dense_data.real,dense_data.imag)).T
    Y = np.concatenate((dense_data.real,dense_data.imag)).T

    X_tensor = torch.tensor(X,dtype=torch.float)
    Y_tensor = torch.tensor(Y,dtype=torch.float)
    dataset = TensorDataset(X_tensor,Y_tensor)

    dataloader = DataLoader(dataset,batch_size = 500,shuffle = True, )
    variance = np.var(Y)
    return dataloader, variance