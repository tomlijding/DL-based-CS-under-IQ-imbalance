from dataclasses import dataclass
import numpy as np
import scipy as sp

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