import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import use
import scipy as sp
from torch import nn
import torch

from src.data_generation import build_dataset
from src.eval import load_pretrained_models, evaluate_pretrained_models
from src.algorithms import omp, psomp, find_x_xi
from src.utils import DataConfig,generate_sensing_matrix, apply_iq_imbalance

# Bugs:
# - In visualization, we calculate and IRR ratio of infinity for 0% imbalance, which is incorrect. Need to handle 0% case separately.
# - Plotting bugs. Namely with visualization of the PSOMP and OMP models alongside the learned models.

config = DataConfig(dataset_size = 1,
    vector_size= 100,
    max_amplitude= 100,
    min_sparsity= 5,
    max_sparsity= 7)
noise_level= 1
omp_epsilon= 1
omp_max_iterations= 10
sensing_matrix_rows= 50
alg= "omp",  # Options: "omp", "psomp", "ml"
model_path= "models/sparse_recovery_model.pth"

noise_levels = [2, 5, 8, 11, 14, 17, 20]
sensing_sizes = [5, 10, 20, 30, 40, 50]

noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models = load_pretrained_models()
all_imbalanced_losses, all_measurement_losses, SNR, IRR_ratios, measurement_sizes, all_noisy_losses = evaluate_pretrained_models(noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models)
loss_fn = nn.MSELoss()
OMP_noisy_losses = []
PSOMP_noisy_losses = []
for noise_level in noise_levels:
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_matrix_rows,config.vector_size)
    # First generate the output
    y = Phi @ x
    y = y + np.random.normal(0, variance, size=y.shape)
    x_hat_omp = omp(Phi,y,omp_epsilon,omp_max_iterations)
    z_hat_psomp = psomp(Phi,y, config.max_sparsity)
    x_hat_psomp, xi_hat_psomp = find_x_xi(z_hat_psomp)
    DFT = sp.linalg.dft(config.vector_size)/np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_NMSE = sum((h-h_hat_omp)**2)/sum(h**2)
    OMP_noisy_losses.append(OMP_NMSE)
    PSOMP_NMSE = sum((h-h_hat_psomp)**2)/sum(h**2)
    PSOMP_noisy_losses.append(PSOMP_NMSE)

OMP_imbalanced_losses = []
PSOMP_imbalanced_losses = []
for IRR_ratio in IRR_ratios:
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_matrix_rows, config.vector_size)
    # First generate the output
    y = Phi @ x
    y = apply_iq_imbalance(y, IRR_ratio)[sensing_matrix_rows:]
    x_hat_omp = omp(Phi, y, omp_epsilon, omp_max_iterations)
    z_hat_psomp = psomp(Phi,y, config.max_sparsity)
    x_hat_psomp, xi_hat_psomp = find_x_xi(z_hat_psomp)
    DFT = sp.linalg.dft(config.vector_size) / np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_NMSE = sum((h - h_hat_omp) ** 2)/sum(h**2)
    OMP_imbalanced_losses.append(OMP_NMSE)
    PSOMP_NMSE = sum((h - h_hat_psomp) ** 2)/sum(h**2)
    PSOMP_imbalanced_losses.append(PSOMP_NMSE)

OMP_sensing_size_losses = []
PSOMP_sensing_size_losses = []
for sensing_size in sensing_sizes:
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_size, config.vector_size)
    # First generate the output
    y = Phi @ x
    x_hat_omp = omp(Phi, y, omp_epsilon, omp_max_iterations)
    z_hat_psomp = psomp(Phi,y, config.max_sparsity)
    x_hat_psomp, xi_hat_psomp = find_x_xi(z_hat_psomp)
    DFT = sp.linalg.dft(config.vector_size) / np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_NMSE = sum((h - h_hat_omp) ** 2)/sum(h**2)
    OMP_sensing_size_losses.append(OMP_NMSE)
    PSOMP_NMSE = sum((h - h_hat_psomp) ** 2)/sum(h**2)
    PSOMP_sensing_size_losses.append(PSOMP_NMSE)

plt.style.use('bmh')
fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))

ax1.plot(SNR.keys(), all_noisy_losses[1], marker="o")
ax1.plot(SNR.keys(), OMP_noisy_losses, marker="o")
# ax1.plot(SNR.keys(), PSOMP_noisy_losses, marker="o")
ax1.set_xlabel("SNR $(dB)$")
ax1.set_ylabel("NMSE")
ax1.set_title("Noisy Model Performance")
ax1.grid(True)
# ax1.set_yscale("log")
ax1.legend(["Auto-encoder", "OMP", "PSOMP"])

ax2.plot(IRR_ratios.values(), all_imbalanced_losses[1], marker='s')
ax2.plot(IRR_ratios.values(), OMP_imbalanced_losses, marker="s")
# ax2.plot(IRR_ratios.values(), PSOMP_imbalanced_losses, marker="s")
ax2.set_xlabel("IRR $(dB)$")
ax2.set_title("IQ Imbalanced Model Performance")
ax2.legend(["Auto-encoder", "OMP", "PSOMP"])
ax2.grid(True)
# ax2.set_yscale("log")

ax3.plot(measurement_sizes, all_measurement_losses[1], marker='^')
ax3.plot(measurement_sizes, OMP_sensing_size_losses, marker="^")
# ax3.plot(measurement_sizes, PSOMP_sensing_size_losses, marker="^")
ax3.set_xlabel("Measurement Dimension")
ax3.set_title("Measurement Model Performance")
ax3.legend(["Auto-encoder", "OMP", "PSOMP"])
ax3.grid(True)
# ax3.set_yscale("log")

plt.show()
