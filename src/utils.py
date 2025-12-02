import numpy as np
import scipy as sp
import torch
from torch import nn
from dataclasses import dataclass

""""
Dataclass Configurations
"""

@dataclass
class Config:
    dataset_size: int = 10000
    vector_size: int = 100
    max_amplitude: int = 100
    min_sparsity: int = 7
    max_sparsity: int = 9
    noise_level: float = 0.01
    omp_epsilon: float = 1e-6
    omp_max_iterations: int = 50
    sensing_matrix_rows: int = 50
    alg: str = "omp"  # Options: "omp", "psomp", "ml"
    model_path: str = "models/sparse_recovery_model.pth"

@dataclass
class DataConfig:
    vector_size: int = 100
    dataset_size: int = 10000
    max_amplitude: int = 100
    min_sparsity: int = 3
    max_sparsity: int = 7


"""
Deep Learning Utilities
"""

def validateModels(dataloader,models,loss_fn,signal_variance=133):
    models_losses = []
    with torch.no_grad():
        for model in models:
            current_model_losses = []
            model.eval()
            for batch in dataloader:
                inputs, targets = batch  # Unpack the tuple
                output = model(inputs)
                loss = loss_fn(output, targets)
                current_model_losses.append(loss.item())
            models_losses.append(np.average(current_model_losses))

    models_losses = np.array(models_losses)
    normalized_models_losses = models_losses/signal_variance
    return normalized_models_losses,models_losses

def discreteLossPoly(qweights,scaleFactor):
    loss = 0
    pi = torch.tensor(math.pi)
    # Note that we need to flatten the weights so that our iteration does not result in us iterating over the rows instead of the weights
    qVec = qweights.flatten()
    # Efficient implementation of the loss function, by doing vector operations, saves a lot of time in training
    loss += torch.linalg.vector_norm(qVec*(qVec-1/2*pi)*(qVec-1*pi)*(qVec+pi)*(qVec+1/2*pi),1)
    loss = loss*scaleFactor # Scale the resulting loss
    return loss


def mapToDiscreteValues(weights, discrete_values):
    # Input is a tensor (possibly a matrix) of weights, and a np array of discrete values
    discrete_values = discrete_values.flatten()
    weights_np = weights.detach().cpu().numpy()  # Convert to numpy array
    shape = weights_np.shape
    weights_vector = np.reshape(weights_np, (-1,
                                             1))  # flatten the matrix to a vector such that subtracting from the discrete values results in a matrix!

    # Create a matrix of distances, then make a vector of indices from this matrix. Each value of the vector is the index of the closest discrete value
    distances = np.abs(weights_vector - discrete_values)
    indices = np.argmin(distances, 1)

    # Map the weights to the closest discrete values and reshape into original matrix, and turn into a nn.Parameter object
    mappedWeights = discrete_values[indices]
    mappedWeights = np.reshape(mappedWeights, shape)
    mappedWeights = np.float32(mappedWeights)  # Notice we map it to a float because that is what is used for our model
    mappedWeights = nn.Parameter(torch.from_numpy(mappedWeights))
    return mappedWeights


"""
Various Utility Functions
"""

def calc_IRR_ratios(imb_percentage_list):
    IRR_ratios = {}
    for level in imb_percentage_list:
        b = 1 - (0.2 * level)
        d = level * np.pi / 8
        r = 0.5 * (1 + b * np.exp(1j * d))
        IRR_ratio = (np.abs(r) ** 2) / (np.abs(1 - r) ** 2)
        IRR_ratios[level] = 10 * np.log10(IRR_ratio)

        return IRR_ratios

"""
OMP/PSOMP
"""

def generate_sensing_matrix(m, n):
    """
    Function to generate a random sensing matrix.
    
    Parameters:
    m : int
        Number of rows (measurements).
    n : int
        Number of columns (signal dimension).
        
    Returns:
    Phi : np.ndarray
        The generated sensing matrix (size: m x n).
    """
    #DFT = sp.linalg.dft(n)/np.sqrt(n)
    A = np.random.randn(m, n)
    Phi = A# @ DFT
    Phi = Phi/ np.linalg.norm(Phi, axis=0, keepdims=True)
    return Phi

def apply_iq_imbalance(x,xi):
    """
    Function which applies IQ imbalance to a given signal.
    Parameters:
        x : np.ndarray
            Input signal (size: n x 1).
        xi : float
            IQ imbalance parameter.
            
        Returns:
        y : np.ndarray
            Signal after applying IQ imbalance (size: n x 1).
    """
    z_1 = xi*x
    z_2 = (1 - np.conj(xi))*np.conj(x)
    z = np.concatenate([z_1,z_2])
    return z.reshape(-1,1)

def generate_random_phase_matrix(m :int ,n : int):
    """
    Function to generate a random phase matrix.
    
    Parameters:
    m : int
        Number of rows.
    n: int
        Number of columns.
        
    Returns:
    P : np.ndarray
        The generated random phase matrix (size: m x n).
    """

    phase_matrix = np.exp(1j *np.random.uniform(-np.pi,np.pi,size=(m,n)))/np.sqrt(n)
    
    return phase_matrix

def unitary_dft(n : int):
    """
    Function to generate a unitary DFT matrix.
    
    Parameters:
    n : int
        Size of the DFT matrix.
        
    Returns:
    DFT : np.ndarray
        The generated unitary DFT matrix (size: n x n).
    """
    DFT = sp.linalg.dft(n)/np.sqrt(n)
    return DFT

def iq_imbalanced_measurement(A : np.ndarray, x : np.ndarray, xi : complex, noise_level : float = 0.0):
    """
    Function to obtain IQ imbalanced measurements of a signal.
    
    Parameters:
    F: np.ndarray
        Sensing matrix (random phases)(size: m x n).
    x : np.ndarray
        Original signal (size: n x 1).
    xi : complex
        IQ imbalance parameter.
    noise_level : float
        Standard deviation of the Gaussian noise to be added.
        
    Returns:
    y : np.ndarray
        IQ imbalanced measurements (size: m x 1).
    """
    y = xi*A@x + (1 - np.conj(xi))*np.conjugate(A)@np.conjugate(x)  # IQ imbalanced measurements
    if noise_level > 0:
        noise = noise_level * (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape))
        y += noise  # Add noise
    return y