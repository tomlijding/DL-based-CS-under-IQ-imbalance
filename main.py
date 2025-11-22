import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
from torch import nn

from data import Generate_Dataloader
from models import (LearnedAutoencoderWithIQImbalance,
                    LearnedAutoencoderWithVarIQImbalance,
                    LearnedAutoencoder,
                    trainModelsForDiscreteSet)
from plotting import visualizeReconstruction, plotting, plot_losses
from pretrained_models import load_pretrained_models, evaluate_pretrained_models
from utils import  discreteLossPoly, mapToDiscreteValues, validateModels

# Build a standard dataset
max_amplitude = 100
min_sparsity = 7
max_sparsity = 9
vector_size = 100
data_set_size = 10000
dataloader, signal_variance = Generate_Dataloader(max_amplitude,min_sparsity,max_sparsity,vector_size,data_set_size)

noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models = load_pretrained_models()
evaluate_pretrained_models(noisy_pretrained_models, imbalanced_pretrained_models, measurement_pretrained_models)

vector_size = 100
encoding_dim = 50
hidden_dims = np.array([60,80])
discrete_autoencoder_model = LearnedAutoencoder(vector_size,encoding_dim,hidden_dims)
optimizer = torch.optim.Adam(discrete_autoencoder_model.parameters(), lr=1E-3, betas=(0.9,0.999))
MSELossfn = nn.MSELoss()
scaleFactor = 0.05 # Hyperparameter, setting this too high causes the problem to not converge to low loss, due to the problem converging to discrete values too early, setting too low causes the
# values to not converge to discrete values. Empirical testing showed that scaleFactor of 0.05 was nice

# Training loop
losses = []
lowest_loss = float("inf")
for epoch in range(5000):
    for batch in dataloader:
        inputs, targets = batch  # Unpack the tuple
        optimizer.zero_grad()
        output = discrete_autoencoder_model(inputs)
        qweights = discrete_autoencoder_model.encoder.q_values
        loss = discreteLossPoly(qweights,scaleFactor) + MSELossfn(output,targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if loss < lowest_loss:
        lowest_loss = loss.item()
        early_stopping_counter = 0
        best_model = discrete_autoencoder_model
    else:
        early_stopping_counter += 1
        if early_stopping_counter > 100:
            discrete_autoencoder_model = best_model
            print(f"stopped early after {epoch+1} epochs, with a loss of: {lowest_loss}")
            break

    print(f"Epoch {epoch+1}, Loss: {lowest_loss:.6f}")

plt.plot(losses)
plt.show()

qweights = discrete_autoencoder_model.encoder.q_values
discrete_values = np.array([-np.pi, -0.5*np.pi,0,0.5*np.pi,np.pi])
mapped_q_weights = mapToDiscreteValues(qweights,discrete_values)


mapped_discrete_autoencoder_model = discrete_autoencoder_model
mapped_discrete_autoencoder_model.encoder.q_values = mapped_q_weights

discrete_model = LearnedAutoencoder(vector_size,encoding_dim,hidden_dims)
# Load the state dictionary
discrete_model.load_state_dict(torch.load("Models/discrete_models/discrete_model.pt"))

visualizeReconstruction(discrete_model,max_amplitude,min_sparsity,max_sparsity,vector_size)

discrete_model_losses = []

dataloader_val, signal_variance = Generate_Dataloader(max_amplitude, min_sparsity, max_sparsity, vector_size, data_set_size)

loss_fn = nn.MSELoss()

# We need to put it in an array for our function to work!
discrete_models = [discrete_model]

normalized_loss, unnormalized_loss = validateModels(dataloader_val,discrete_models,loss_fn)

print(f"Normalized loss is:{normalized_loss}, Unnormalized loss is:{unnormalized_loss}")

# Training just one model for illustration purposes
encoding_dims_list = [30]
SNR_list = [np.inf]
imb_percentage_list = [0]

discrete_models = trainModelsForDiscreteSet(dataloader,SNR_list,imb_percentage_list,encoding_dims_list,scale_factor=0.01)

encoding_dims_list = [50, 40, 30, 20, 10, 5, 50, 50, 50, 50,50,50,50,50,50,50,50,50,50]
SNR_list = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 20, 17, 14, 11, 8, 5, 2, np.inf, np.inf,np.inf, np.inf,np.inf, np.inf]
imb_percentage_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0.04, 0.1, 0.3, 0.6, 1]

discrete_models = []

for iter,__ in enumerate(SNR_list):
    abs_noise_ratio = 10**(SNR_list[iter]/10)
    variance = signal_variance/abs_noise_ratio
    b = 1 - (0.2 * imb_percentage_list[iter])
    d = imb_percentage_list[iter] * np.pi/8
    hidden_dims = np.array([60,80])
    discrete_models.append(LearnedAutoencoderWithIQImbalance(vector_size,encoding_dims_list[iter],hidden_dims,b,d,variance))
    discrete_models[iter].load_state_dict(torch.load(
        f'Models/discrete_models/discrete_model_SNR{SNR_list[iter]}_IRR{imb_percentage_list[iter]}_enc{encoding_dims_list[iter]}.pt', weights_only=True))

loss_fn = nn.MSELoss()
normalized_losses, unnormalized_losses = validateModels(dataloader_val,discrete_models,loss_fn)
visualizeReconstruction(discrete_models[2])

plotting(imb_percentage_list, SNR_list, normalized_losses, encoding_dims_list)
plt.show()

# Create an array of 100 numbers (0 to 99)
numbers = list(range(100))
RIC = {}

# Generate all possible 3-length combinations
for model in discrete_models:
    max_RIC = 0
    # Get q-values and create the complex matrix W
    qvalues = model.encoder.q_values.data.numpy()
    W = np.e ** (1j * qvalues)
    DFT = sp.linalg.dft(vector_size) / np.sqrt(vector_size)
    W = W @ DFT
    # Normalize each column so they have unit norm
    col_norms = np.linalg.norm(W, axis=0)
    diag_norm_matrix = np.diag(col_norms)
    W_normalized = W @ np.linalg.inv(diag_norm_matrix)

    for combo in itertools.combinations(numbers, 3):
        # Select the columns specified by the combination
        W_cols = W_normalized[:, combo]
        mod_mat = W_cols.T.conj() @ W_cols - np.eye(3)

        # Compute eigenvalues of the Gram matrix
        eig_vals, _ = np.linalg.eig(mod_mat)
        eigenvalues = np.abs(eig_vals)  # They should be real and close to 1

        temp_RIC = np.max(eigenvalues)

        if temp_RIC > max_RIC:
            max_RIC = temp_RIC

    RIC[model] = max_RIC

print(RIC.values())
models = discrete_models

mu = {}
for model in discrete_models:
    qvalues = model.encoder.q_values.data.numpy()
    W = np.e**(1j * qvalues)
    DFT = sp.linalg.dft(vector_size)/np.sqrt(vector_size)
    A = W@DFT
    # Normalize each column so they have unit norm
    col_norms = np.linalg.norm(A, axis=0)
    diag_norm_matrix = np.diag(col_norms)
    A_normalized = A @ np.linalg.inv(diag_norm_matrix)
    A_dotprod = np.abs(A_normalized.conj().T@A_normalized)
    A_no_diag = A_dotprod - np.diag(np.diag(A_dotprod))
    mu[model] = np.max(A_no_diag)

print(mu.values())

# Generate the data
max_amplitude = 100
min_sparsity = 7
max_sparsity = 9
vector_size = 100
data_set_size = 10000

dataloader, signal_variance = Generate_Dataloader(max_amplitude, min_sparsity, max_sparsity, vector_size, data_set_size)

# discrete_values = np.array([-np.pi, -0.5*np.pi,0,0.5*np.pi,np.pi])
scale_factor = 0.01
vector_size = 100
encoding_dim = 50
variance = 0
hidden_dims = np.array([60,80])
current_training_model = LearnedAutoencoderWithVarIQImbalance(vector_size,encoding_dim,hidden_dims,variance)
optimizer = torch.optim.Adam(current_training_model.parameters(), lr=1E-3, betas=(0.9,0.999))
MSEloss_fn = nn.MSELoss()

# Training loop
losses = []
lowest_loss = float("inf")
for epoch in range(10000):
    for batch in dataloader:
        b = np.random.uniform(0.8,1)
        d = np.random.uniform(0,np.pi/8)
        current_training_model.b = b
        current_training_model.d = d
        inputs, targets = batch  # Unpack the tuple
        optimizer.zero_grad()
        output = current_training_model(inputs)
        # qweights = current_training_model.encoder.q_values
        loss = MSEloss_fn(output, targets) # + discreteLossPoly(qweights,scale_factor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if loss< lowest_loss:
        lowest_loss = loss
        early_stopping_counter = 0
        best_model = current_training_model
    else:
        early_stopping_counter += 1
        if early_stopping_counter > 100:
            current_training_model = best_model
            print(f"Stopped early after {epoch+1} epochs, with loss of {lowest_loss:.6f}")
            break
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
torch.save(best_model.state_dict(),f'Models/random_imbalanced_models/random_imbalanced_model.pt')

vector_size = 100
encoding_dim = 50
variance = 0
hidden_dims = np.array([60,80])
random_imb_model = LearnedAutoencoderWithVarIQImbalance(vector_size,encoding_dim,hidden_dims,variance)
random_imb_model.load_state_dict(torch.load("Models/random_imbalanced_models/random_imbalanced_model.pt",weights_only=True))

loss_fn = nn.MSELoss()

iq_imbalanced_models = []
imb_percentage_list = [0, 0.04, 0.1, 0.3, 0.6, 1]
IRR_ratios = []
for level in imb_percentage_list:
    b = 1 - (0.2 * level)
    d = level * np.pi / 8
    r = 0.5 * (1 + b * np.exp(1j * d))
    IRR_ratio = (np.abs(r) ** 2) / (np.abs(1 - r) ** 2)
    IRR_ratios.append(10 * np.log10(IRR_ratio))
    random_imb_model.b = b
    random_imb_model.d = d
    iq_imbalanced_models.append(random_imb_model)
    print(random_imb_model.b)
    print(random_imb_model.d)

max_amplitude = 100
min_sparsity = 7
max_sparsity = 9
vector_size = 100
data_set_size = 10000

dataloader_val, signal_variance = Generate_Dataloader(max_amplitude, min_sparsity, max_sparsity, vector_size,
                                                      data_set_size)

normalized_losses, unnormalized_losses = validateModels(dataloader_val, iq_imbalanced_models, loss_fn)

plot_losses(IRR_ratios, normalized_losses)
plt.show()
print(normalized_losses)
print(unnormalized_losses)


