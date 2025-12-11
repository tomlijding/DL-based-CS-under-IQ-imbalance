import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from torch import nn

from data import build_dataset

def plot_several_models(all_imbalanced_losses, all_measurement_losses, SNR, IRR_ratios, measurement_sizes, all_noisy_losses):
    plt.style.use('bmh')
    fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))

    for i in range(4):
        if i == 0:
            noiseless_loss = all_imbalanced_losses[0][0]
            ax1.plot(SNR.keys(), [noiseless_loss for i in SNR.keys()])
            ax2.plot(IRR_ratios.values(), [noiseless_loss for i in IRR_ratios.values()])
            ax3.plot(measurement_sizes, [noiseless_loss for i in measurement_sizes])

        ax1.plot(SNR.keys(), all_noisy_losses[i], marker="o")
        ax1.set_xlabel("SNR $(dB)$")
        ax1.set_ylabel("NMSE")
        ax1.set_title("Noisy Model Performance")
        ax1.grid(True)
        ax1.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30"])

        ax2.plot(IRR_ratios.values(), all_imbalanced_losses[i], marker='s')
        ax2.set_xlabel("IRR $(dB)$")
        ax2.set_title("IQ Imbalanced Model Performance")
        ax2.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30"])
        ax2.grid(True)

        ax3.plot(measurement_sizes, all_measurement_losses[i], marker='^')
        ax3.set_xlabel("Measurement Dimension")
        ax3.set_title("Measurement Model Performance")
        ax3.legend(["baseline model", "sparsity 3-5", "sparsity 5-7", "sparsity 7-9", "sparsity 10-30"])
        ax3.grid(True)

    plt.show()

def visualizeReconstruction(model, max_amplitude=100, min_sparsity=3, max_sparsity=5, vector_size=100):
    h, x = build_dataset(max_amplitude, min_sparsity, max_sparsity, vector_size, 1)

    H = np.concatenate((h.real, h.imag)).T

    H_tensor = torch.tensor(H, dtype=torch.float)

    H_hat = model(H_tensor)

    h_hat = np.array(H_hat.detach())

    h_real, h_imag = np.split(h_hat, 2, 1)
    h_hat = h_real + 1j * h_imag
    h_hat = h_hat.reshape(-1, 1)
    DFT = sp.linalg.dft(vector_size) / np.sqrt(vector_size)
    iDFT = DFT.conj().T

    x_hat = iDFT @ h_hat
    indices = range(len(x_hat))

    plt.vlines(indices, 0, x, linewidth=3)
    plt.vlines(indices, 0, x_hat, colors="orange")

    plt.legend(("x", "x_hat"))

def plotting(imb_percentage_list, SNR_list, normalized_losses, encoding_dims_list):
    # Quick calculation for the sake of plotting
    imb_db_list = []

    for perc in imb_percentage_list[13:19]:
        b = 1 - 0.2 * perc
        d = np.pi / 8 * perc
        r = 0.5 * (1 + b * np.exp(1j * d))
        IRR_abs = np.abs(r) ** 2 / np.abs(1 - r) ** 2
        imb_db_list.append(10 * np.log10(IRR_abs))

    plt.style.use('ggplot')
    fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(18, 6))

    ax1.plot(SNR_list[6:13], normalized_losses[6:13], marker="o", color="g")
    ax2.plot(imb_db_list, normalized_losses[13:20], marker='s', color='b')
    ax3.plot(encoding_dims_list[0:6], normalized_losses[0:6], marker='^', color='r')

    ax1.set_xlabel("SNR $(dB)$")
    ax1.set_ylabel("NMSE")
    ax1.set_title("Noisy Model Performance")
    ax1.grid(True)
    ax1.legend(["Discrete Model"])

    ax2.set_xlabel("IRR $(dB)$")
    ax2.set_title("IQ Imbalanced Model Performance")
    ax2.legend(["Discrete Model"])
    ax2.grid(True)

    ax3.set_xlabel("Measurement Dimension")
    ax3.set_title("Measurement Model Performance")
    ax3.legend(["Discrete Model"])
    ax3.grid(True)
    plt.savefig("Images/discrete_model_performance.pdf", format="pdf", bbox_inches="tight")

def plot_losses(IRR_ratios, normalized_losses):
    fig, ax = plt.subplots()
    ax.plot(IRR_ratios, normalized_losses, '-o')
    ax.set_xlabel('IRR Ratios [-]')
    ax.set_ylabel('NMSE [-]')

    # Turn off the offset notation
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')
