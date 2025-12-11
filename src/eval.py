import numpy as np
import torch
import torch.nn as nn
from src.data_generation import generate_dataloader
from src.algorithms import LearnedAutoencoderWithNoise, LearnedAutoencoderWithIQImbalance
from src.visualization import plot_several_models
from src.utils import DataConfig, calc_IRR_ratios
import matplotlib.pyplot as plt

"""
Model Evaluation Utilities
"""

def load_pretrained_models():
    # Initialize pretrained models
    vector_size = 100
    sparsity_ranges = [(3, 5), (5, 7), (7, 9), (10, 30)]
    measurement_sizes = [5, 10, 20, 30, 40, 50]
    # Define the SNR dictionary for use
    db_list = [2, 5, 8, 11, 14, 17, 20]
    signal_variance = 133  # Found by measuring empirically what the variance of the signal is once transformed from sparse signal
    SNR = {}

    # Then define the absolute values of the SNR ratio
    for db_ratio in db_list:
        SNR[db_ratio] = 10 ** (db_ratio / 10)

    imb_percentage_list = [0, 0.04, 0.1, 0.3, 0.6, 1]
    IRR_ratios = calc_IRR_ratios(imb_percentage_list)

    # empty dicts to store models in
    noisy_pretrained_models = {}
    imbalanced_pretrained_models = {}
    measurement_pretrained_models = {}

    for i, (min_spars, max_spars) in enumerate(sparsity_ranges):

        # Initialize pretrained noisy models
        for db, abs in SNR.items():
            encoding_dim = 50
            variance = signal_variance / abs
            # Initialize model
            hidden_dims = np.array([60, 80])
            noisy_pretrained_models[(i, db)] = LearnedAutoencoderWithNoise(vector_size, encoding_dim, hidden_dims,
                                                                           variance)
            noisy_pretrained_models[(i, db)].load_state_dict(torch.load(
                f"../Models/noisy_models/sparsity_{min_spars}-{max_spars}/noisy_model_{db}_{min_spars}-{max_spars}.pt",
                weights_only=True))

        # Initialize pretrained imbalanced_models
        for level, db in IRR_ratios.items():
            encoding_dim = 50
            variance = 0
            b = 1 - (0.2 * level)
            d = level * np.pi / 8
            # Initialize model
            hidden_dims = np.array([60, 80])
            imbalanced_pretrained_models[(i, level)] = LearnedAutoencoderWithIQImbalance(vector_size, encoding_dim,
                                                                                         hidden_dims, b, d, variance)
            imbalanced_pretrained_models[(i, level)].load_state_dict(torch.load(
                f"../Models/imbalanced_models/sparsity_{min_spars}-{max_spars}/imbalanced_model_{level:.3f}_{min_spars}-{max_spars}.pt",
                weights_only=True))

        # Initialize pretrained measurement models
        for encoding_dim in measurement_sizes:
            variance = signal_variance / SNR[17]
            level = 0.6
            b = 1 - (0.2 * level)
            d = level * np.pi / 8
            # Initialize model
            hidden_dims = np.array([60, 80])
            measurement_pretrained_models[(i, encoding_dim)] = LearnedAutoencoderWithIQImbalance(vector_size,
                                                                                                 encoding_dim,
                                                                                                 hidden_dims, b, d,
                                                                                                 variance)
            measurement_pretrained_models[(i, encoding_dim)].load_state_dict(torch.load(
                f"../Models/measurement_models/sparsity_{min_spars}-{max_spars}/measurement_model_{encoding_dim}_{min_spars}-{max_spars}.pt",
                weights_only=True))
    return noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models


def evaluate_pretrained_models(noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models):
    # Initialize pretrained models
    data_set_size = 10000
    max_amplitude = 100
    vector_size = 100
    sparsity_ranges = [(3, 5), (5, 7), (7, 9), (10, 30)]
    measurement_sizes = [5, 10, 20, 30, 40, 50]
    # Define the SNR dictionary for use
    db_list = [2, 5, 8, 11, 14, 17, 20]
    signal_variance = 133  # Found by measuring empirically what the variance of the signal is once transformed from sparse signal
    SNR = {}

    # Then define the absolute values of the SNR ratio
    for db_ratio in db_list:
        SNR[db_ratio] = 10 ** (db_ratio / 10)

    imb_percentage_list = [0, 0.04, 0.1, 0.3, 0.6, 1]
    IRR_ratios = calc_IRR_ratios(imb_percentage_list)

    all_noisy_losses = []
    all_imbalanced_losses = []
    all_measurement_losses = []

    loss_fn = nn.MSELoss()

    for i, (min_spars, max_spars) in enumerate(sparsity_ranges):
        dataloader_val, signal_variance = generate_dataloader(DataConfig)
        # Evaluate noisy models
        noisy_val_losses = []
        noisy_model_losses = []
        with torch.no_grad():
            for db, abs in SNR.items():
                noisy_model = noisy_pretrained_models[(i, db)]
                noisy_model.eval()
                for batch in dataloader_val:
                    inputs, targets = batch  # Unpack the tuple
                    output = noisy_model(inputs)
                    loss = loss_fn(output, targets)
                    noisy_model_losses.append(loss.item())
                noisy_val_losses.append(np.average(noisy_model_losses))
                noisy_model_losses = []

        noisy_val_losses = np.array(noisy_val_losses)
        normalized_noisy_val_losses = noisy_val_losses/signal_variance
        all_noisy_losses.append(normalized_noisy_val_losses)

        # Evaluate Imbalanced models
        imbalance_model_losses = []
        imbalance_val_losses = []

        with torch.no_grad():
            for level, db in IRR_ratios.items():
                imbalance_model = imbalanced_pretrained_models[(i, level)]
                imbalance_model.eval()
                for batch in dataloader_val:
                    inputs, targets = batch  # Unpack the tuple
                    output = imbalance_model(inputs)
                    loss = loss_fn(output, targets)
                    imbalance_model_losses.append(loss.item())
                imbalance_val_losses.append(np.average(imbalance_model_losses))
                imbalance_model_losses = []

        imbalance_val_losses = np.array(imbalance_val_losses)
        normalized_imbalance_val_losses = imbalance_val_losses/signal_variance
        all_imbalanced_losses.append(normalized_imbalance_val_losses)

        # Evaluate models with varying measurement sizes
        measurement_model_losses = []
        measurement_val_losses = []

        with (torch.no_grad()):
            for encoding_dim in measurement_sizes:
                measurement_model = measurement_pretrained_models[(i, encoding_dim)]
                measurement_model.eval()
                for batch in dataloader_val:
                    inputs, targets = batch  # Unpack the tuple
                    output = measurement_model(inputs)
                    loss = loss_fn(output, targets)
                    measurement_model_losses.append(loss.item())
                measurement_val_losses.append(np.average(measurement_model_losses))
                measurement_model_losses = []

        measurement_val_losses = np.array(measurement_val_losses)
        normalized_measurement_val_losses = measurement_val_losses/signal_variance
        all_measurement_losses.append(normalized_measurement_val_losses)

    plot_several_models(all_imbalanced_losses, all_measurement_losses, SNR, IRR_ratios, measurement_sizes, all_noisy_losses)
    plt.show()

    return all_imbalanced_losses, all_measurement_losses, SNR, IRR_ratios, measurement_sizes, all_noisy_losses
