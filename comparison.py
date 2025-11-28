import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from data import build_dataset, DataConfig
from pretrained_models import load_pretrained_models, evaluate_pretrained_models
from utils import generate_sensing_matrix, apply_iq_imbalance
from models import omp, psomp


config = DataConfig(dataset_size = 1,
    vector_size= 100,
    max_amplitude= 100,
    min_sparsity= 5,
    max_sparsity= 10)
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

print("training (PS)OMP model")
OMP_noisy_losses = []
PSOMP_noisy_losses = []
for noise_level in noise_levels:
    print(f"Printing models with {noise_level}dB of noise")
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_matrix_rows,config.vector_size)
    # First generate the output
    y = Phi @ x
    y = y + np.random.normal(0, variance, size=y.shape)
    x_hat_omp = omp(Phi,y,omp_epsilon,omp_max_iterations)
    x_hat_psomp = psomp(Phi,y, config.max_sparsity)
    DFT = sp.linalg.dft(config.vector_size)/np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_MSE = sum((x-x_hat_omp)**2)/len(y)
    OMP_noisy_losses.append(OMP_noisy_losses)
    PSOMP_MSE = sum((x-x_hat_psomp)**2)/len(y)
    PSOMP_noisy_losses.append(PSOMP_noisy_losses)

OMP_imbalanced_losses = []
PSOMP_imbalanced_losses = []
for IRR_ratio in IRR_ratios:
    print(f"Printing models with {IRR_ratio} IRR ratio")
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_matrix_rows, config.vector_size)
    # First generate the output
    y = Phi @ x
    y = apply_iq_imbalance(y, IRR_ratio)[sensing_matrix_rows:]
    x_hat_omp = omp(Phi, y, omp_epsilon, omp_max_iterations)
    x_hat_psomp = psomp(Phi,y, config.max_sparsity)
    DFT = sp.linalg.dft(config.vector_size) / np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_MSE = sum((x - x_hat_omp) ** 2)
    OMP_imbalanced_losses.append(OMP_MSE)
    PSOMP_MSE = sum((x - x_hat_psomp) ** 2)
    PSOMP_imbalanced_losses.append(PSOMP_MSE)

OMP_sensing_size_losses = []
PSOMP_sensing_size_losses = []
for sensing_size in sensing_sizes:
    print(f"Training models with sensing matrix with {sensing_size} columns")
    variance = 133 / (10 ** (noise_level / 10))
    h, x = build_dataset(config)
    Phi = generate_sensing_matrix(sensing_size, config.vector_size)
    # First generate the output
    y = Phi @ x
    x_hat_omp = omp(Phi, y, omp_epsilon, omp_max_iterations)
    x_hat_psomp = psomp(Phi,y, config.max_sparsity)
    DFT = sp.linalg.dft(config.vector_size) / np.sqrt(config.vector_size)
    h_hat_omp = DFT @ x_hat_omp
    h_hat_psomp = DFT @ x_hat_psomp
    indices = range(len(x_hat_omp))

    OMP_MSE = sum((x - x_hat_omp) ** 2)
    OMP_imbalanced_losses.append(OMP_MSE)
    PSOMP_MSE = sum((x - x_hat_psomp) ** 2)
    PSOMP_imbalanced_losses.append(PSOMP_MSE)

print("model training complete")
plt.style.use('ggplot')
fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))

for i in range(4):
    if i == 0:
        noiseless_loss = all_imbalanced_losses[0][0]
        ax1.plot(SNR.keys(), [noiseless_loss for i in SNR.keys()])
        ax2.plot(IRR_ratios.values(), [noiseless_loss for i in IRR_ratios.values()])
        ax3.plot(measurement_sizes, [noiseless_loss for i in measurement_sizes])

    ax1.plot(SNR.keys(), all_noisy_losses[i], marker="o")
    ax1.plot(SNR.keys(), OMP_noisy_losses, marker="o")
    ax1.plot(SNR.keys(), PSOMP_noisy_losses, marker="o")
    ax1.set_xlabel("SNR $(dB)$")
    ax1.set_ylabel("NMSE")
    ax1.set_title("Noisy Model Performance")
    ax1.grid(True)
    ax1.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30", "OMP", "PSOMP"])

    ax2.plot(IRR_ratios.values(), all_imbalanced_losses[i], marker='s')
    ax2.plot(IRR_ratios.values(), OMP_imbalanced_losses, marker="s")
    ax2.plot(IRR_ratios.values(), PSOMP_imbalanced_losses, marker="s")
    ax2.set_xlabel("IRR $(dB)$")
    ax2.set_title("IQ Imbalanced Model Performance")
    ax2.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30", "OMP", "PSOMP"])
    ax2.grid(True)

    ax3.plot(measurement_sizes, all_measurement_losses[i], marker='^')
    ax3.plot(measurement_sizes, OMP_sensing_size_losses, marker="^")
    ax3.plot(measurement_sizes, PSOMP_sensing_size_losses, marker="^")
    ax3.set_xlabel("Measurement Dimension")
    ax3.set_title("Measurement Model Performance")
    ax3.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30", "OMP", "PSOMP"])
    ax3.grid(True)
plt.show()
