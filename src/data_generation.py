import numpy as np
import scipy as sp
from src.utils import Config
import random

def build_dataset(config: Config):
    """
    Function to build a dataset of sparse and dense signals.
    Parameters:
    config : Config
        Configuration object containing parameters for dataset generation.
    Returns:
    dense_data : np.ndarray
        The dense signal dataset (size: vector_size x data_set_size).
    sparse_data : np.ndarray
        The sparse signal dataset (size: vector_size x data_set_size).
    """
    # Fetch configuration parameters
    vector_size = config.vector_size
    data_set_size = config.dataset_size
    max_amplitude = config.max_amplitude
    min_sparsity = config.min_sparsity
    max_sparsity = config.max_sparsity

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

def generate_sparse_vector(sparsity: int, vector_size: int, max_amplitude: int):
    """
    Function to generate a single sparse vector.
    Parameters:
    sparsity : int
        The number of non-zero elements in the sparse vector.
    vector_size : int
        The size of the sparse vector.
    max_amplitude : int
        The maximum amplitude for the non-zero elements.
        
    Returns:
    x : np.ndarray
        The generated sparse vector (size: vector_size x 1).
    """
    x = np.zeros((vector_size, 1), dtype=float)  # Ensure float type
    indices = random.sample(range(vector_size), sparsity)
    amps = np.random.uniform(-max_amplitude, max_amplitude, sparsity)  # Use negative and positive values
    x[indices, 0] = amps
    return x