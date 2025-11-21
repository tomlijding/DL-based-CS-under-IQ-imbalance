from src.data_generation import build_dataset
import numpy as np
import torch
from src.utils import Config
import scipy as sp
import matplotlib.pyplot as plt
from src.algorithms import omp
from src.utils import generate_sensing_matrix

def visualize_reconstruction(config: Config, model = None):
    """
    Function to visualize the reconstruction of a sparse signal using different algorithms.
    Parameters:
    config : Config
        Configuration object containing parameters for the algorithms.
    model : torch.nn.Module, optional
        Pre-trained model for ML-based reconstruction (default is None).
    max_amplitude : int
        Maximum amplitude of the sparse signal.
    min_sparsity : int
        Minimum sparsity level of the sparse signal.
    max_sparsity : int
        Maximum sparsity level of the sparse signal.
    vector_size : int
        Size of the sparse signal vector.
    """
    vector_size = config.vector_size
    # h is the dense signal, x is the sparse signal
    h, x = build_dataset(config)

    if Config.alg == "ml":
        H = np.concatenate((h.real,h.imag)).T

        H_tensor = torch.tensor(H,dtype=torch.float)
        if model is None:
            raise ValueError("Model is not loaded.")
        H_hat = model(H_tensor)

        h_hat = np.array(H_hat.detach())

        h_real,h_imag = np.split(h_hat,2,1)
        h_hat = h_real + 1j*h_imag
        h_hat = h_hat.reshape(-1,1)
        DFT = sp.linalg.dft(vector_size)/np.sqrt(vector_size)
        iDFT = DFT.conj().T

        x_hat = iDFT@h_hat
    
    elif Config.alg == "omp":
        A = generate_sensing_matrix(config.sensing_matrix_rows,config.vector_size)
        DFT = sp.linalg.dft(vector_size)/np.sqrt(vector_size)
        # First generate the output
        y = A @ h
        x_hat = omp(A@DFT,y,config.omp_epsilon,config.omp_max_iterations)

    else:
        raise ValueError(f"Algorithm {Config.alg} not recognized.")

    indices = range(len(x_hat))
    plt.vlines(indices,0,x,linewidth=3)
    plt.vlines(indices,0,x_hat,colors="orange")

    plt.legend(("x","x_hat"))