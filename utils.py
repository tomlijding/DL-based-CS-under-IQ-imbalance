import math

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update(plt.rcParamsDefault)
import torch
from torch import nn

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

def calc_IRR_ratios(imb_percentage_list):
    IRR_ratios = {}
    for level in imb_percentage_list:
        b = 1 - (0.2 * level)
        d = level * np.pi / 8
        r = 0.5 * (1 + b * np.exp(1j * d))
        IRR_ratio = (np.abs(r) ** 2) / (np.abs(1 - r) ** 2)
        IRR_ratios[level] = 10 * np.log10(IRR_ratio)

        return IRR_ratios