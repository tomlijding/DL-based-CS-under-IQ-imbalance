import numpy as np
import torch
from torch import nn
from torch.nn import init

from src.utils import discreteLossPoly, mapToDiscreteValues

"""
Deep Learning
"""

def complex_xavier_init(tensor_real, tensor_imag, gain=1.0):
    """
    Apply Xavier initialization (using uniform variant) to both real and imaginary parts
    """
    init.xavier_uniform_(tensor_real, gain=gain)
    init.xavier_uniform_(tensor_imag, gain=gain)


class ComplexUnitModulus(nn.Module):
    # This class serves as the encoder layer. We are restricted to values which are of the form e^jq where q are trainable parameters
    # Notice that the input and output dimensions are half of what the actual vector size is! Because it is a complex value, our dimensions are twice as long
    def __init__(self, input_dim, output_dim):
        super(ComplexUnitModulus, self).__init__()
        # Here we create the q-values of our unitary matrix. These are the parameters we are training such that each entry of our complex matrix to encode our data is |F_ij| = 1
        self.q_values = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        # Compute unitary weights dynamically in each forward pass
        W_real = torch.cos(self.q_values)
        W_imag = torch.sin(self.q_values)
        W_top = torch.cat([W_real, -W_imag], dim=1)  # [W_real, -W_imag]
        W_bottom = torch.cat([W_imag, W_real], dim=1)  # [W_imag, W_real]
        W_total = torch.cat([W_top, W_bottom], dim=0)  # Stack rows to form the full matrix
        out = torch.matmul(x, W_total.T)
        return out


class ComplexLinear(nn.Module):
    # This custom layer was found to work less well than a regular linear layer, probably because we put restrictions on the network allowing it to be less expressive.
    # Notice that the input and output dimensions are half of what the actual vector size is! Because it is a complex value, our dimensions are twice as long. This gets fixed because we make the matrix
    # W_total which multiplies [x_real;x_imag] and returns [y_real;y_imag]
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        # Here we create the complex matrix W
        # self.W_real = nn.Parameter(torch.randn(output_dim,input_dim))# eye(input_dim))
        # self.W_imag = nn.Parameter(torch.randn(output_dim,input_dim)) #zeros((input_dim,output_dim)))

        self.W_real = nn.Parameter(torch.empty(output_dim, input_dim))
        self.W_imag = nn.Parameter(torch.empty(output_dim, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize both the real and imaginary parts using Xavier initialization.
        complex_xavier_init(self.W_real, self.W_imag)

    def forward(self, x):
        # Compute unitary weights dynamically in each forward pass
        W_real = self.W_real
        W_imag = self.W_imag
        W_top = torch.cat([W_real, -W_imag], dim=1)  # [W_real, -W_imag]
        W_bottom = torch.cat([W_imag, W_real], dim=1)  # [W_imag, W_real]
        W_total = torch.cat([W_top, W_bottom], dim=0)  # Stack rows to form the full matrix
        out = torch.matmul(x, W_total.T)
        return out


class LearnedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims):
        super(LearnedAutoencoder, self).__init__()

        self.encoder = ComplexUnitModulus(input_dim, encoding_dim)
        layers = []
        prev_dim = encoding_dim * 2
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim * 2))
            layers.append(nn.ReLU())
            prev_dim = dim * 2
        self.decoder = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, input_dim * 2)
        )

    def forward(self, x):
        encoder_out = self.encoder(x)

        return self.decoder(encoder_out)


class LearnedAutoencoderWithNoise(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims, variance):
        super(LearnedAutoencoderWithNoise, self).__init__()
        self.variance = variance
        self.encoder = ComplexUnitModulus(input_dim, encoding_dim)
        self.encoding_dim = encoding_dim
        layers = []
        prev_dim = encoding_dim * 2
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim * 2))
            layers.append(nn.ReLU())
            prev_dim = dim * 2
        self.decoder = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, input_dim * 2)
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        noise_np = np.random.normal(0, self.variance, size=self.encoding_dim * 2)
        noise = torch.tensor(noise_np, dtype=torch.float)
        noisy_y = encoder_out + noise
        return self.decoder(noisy_y)


# It should be noted that all previous autoencoders are less general versions of this neural network architecture.
# If we set the IQ imbalance to 0 and the SNR to inf(), then we get previous architectures
class LearnedAutoencoderWithIQImbalance(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims, b, d, variance):
        super(LearnedAutoencoderWithIQImbalance, self).__init__()
        self.encoder = ComplexUnitModulus(input_dim, encoding_dim)
        self.encoding_dim = encoding_dim
        self.variance = variance
        self.r = torch.tensor(0.5 * (1 + b * np.exp(1j * d)), dtype=torch.complex64)
        layers = []
        prev_dim = encoding_dim * 2
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim * 2))
            layers.append(nn.ReLU())
            prev_dim = dim * 2
        self.decoder = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, input_dim * 2)
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        y_real = encoder_out[:, :self.encoding_dim]
        y_imag = encoder_out[:, self.encoding_dim:]
        y = torch.complex(y_real, y_imag)
        yiq = self.r * y + (1 - self.r.conj()) * (y.conj())
        yiqr = yiq.real
        yiqi = yiq.imag
        yiqstack = torch.cat((yiqr, yiqi), dim=1)
        noise_np = np.random.normal(0, self.variance, size=self.encoding_dim * 2)
        noise_tensor = torch.tensor(noise_np, dtype=torch.float)
        y_iq_stack_noisy = yiqstack + noise_tensor
        return self.decoder(y_iq_stack_noisy)

def trainModels(dataloader,SNR_values,imb_percentages,encoding_dims,epochs,signal_variance = 133,hidden_dims=[60,80], input_dim=100):
    # Function takes as inputs:
    # dataloader: The dataloader object of the training data set
    # SNR_values: Signal to noise ratios
    # imb_percentages: imbalance percentages
    # encoding_dims: Encoding dimensions
    # epochs: Maximum amount of epochs we want the model to run for
    # signal_variance: The variance of the original signal
    # hidden_dims: Hidden dimensions for the neural network
    # Returns the best model with corresponding MSELoss
    models = []
    for model_num,(SNR,imb_percentage,encoding_dim) in enumerate(zip(SNR_values,imb_percentages,encoding_dims)):
        abs_noise_ratio = 10**(SNR/10)
        variance = signal_variance/abs_noise_ratio
        b = 1 - (0.2 * imb_percentage)
        d = imb_percentage * np.pi/8
        hidden_dims = np.array([60,80])
        current_training_model = LearnedAutoencoderWithIQImbalance(input_dim,encoding_dim,hidden_dims,b,d,variance)
        optimizer = torch.optim.Adam(current_training_model.parameters(), lr=1E-3, betas=(0.9,0.999))
        MSEloss_fn = nn.MSELoss()

        # Training loop
        losses = []
        lowest_loss = float("inf")
        for epoch in range(epochs):
            for batch in dataloader:
                inputs, targets = batch  # Unpack the tuple
                optimizer.zero_grad()
                output = current_training_model(inputs)
                loss = MSEloss_fn(output, targets)
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
            print(f"SNR:{SNR}, Imbalance Percentage:{imb_percentage}, Encoding dimension:{encoding_dim}, Epoch {epoch+1}, Loss: {loss.item():.6f}")
        models.append(best_model)
        losses.append(lowest_loss)
    return models, losses

def trainModelsForDiscreteSet(dataloader,SNR_values,imb_percentages,encoding_dims,signal_variance = 133,hidden_dims=[60,80], input_dim=100, scale_factor=0.05):
    # Function takes as inputs:
    # dataloader: The dataloader object of the training data set
    # SNR_values: Signal to noise ratios
    # imb_percentages: imbalance percentages
    # encoding_dims: Encoding dimensions
    # signal_variance: The variance of the original signal
    # hidden_dims: Hidden dimensions for the neural network
    # scale_factor: Hyperparameter for the discretization step
    models = []
    discrete_values = np.array([-np.pi, -0.5*np.pi,0,0.5*np.pi,np.pi])
    for model_num,(SNR,imb_percentage,encoding_dim) in enumerate(zip(SNR_values,imb_percentages,encoding_dims)):
        abs_noise_ratio = 10**(SNR/10)
        variance = signal_variance/abs_noise_ratio
        b = 1 - (0.2 * imb_percentage)
        d = imb_percentage * np.pi/8
        hidden_dims = np.array([60,80])
        current_training_model = LearnedAutoencoderWithIQImbalance(input_dim,encoding_dim,hidden_dims,b,d,variance)
        optimizer = torch.optim.Adam(current_training_model.parameters(), lr=1E-3, betas=(0.9,0.999))
        MSEloss_fn = nn.MSELoss()

        # Training loop
        losses = []
        lowest_loss = float("inf")
        for epoch in range(10000):
            for batch in dataloader:
                inputs, targets = batch  # Unpack the tuple
                optimizer.zero_grad()
                output = current_training_model(inputs)
                qweights = current_training_model.encoder.q_values
                loss = discreteLossPoly(qweights,scale_factor) + MSEloss_fn(output, targets)
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
            print(f"SNR:{SNR}, Imbalance Percentage:{imb_percentage}, Encoding dimension:{encoding_dim}, Epoch {epoch+1}, Loss: {loss.item():.6f}")
        best_qvalues = best_model.encoder.q_values
        mapped_best_qvalues = mapToDiscreteValues(best_qvalues,discrete_values)
        best_model.encoder.q_values = mapped_best_qvalues
        models.append(best_model)
        losses.append(lowest_loss)
    return models

class LearnedAutoencoderWithVarIQImbalance(nn.Module):
    def __init__(self, input_dim, encoding_dim,hidden_dims,variance,b=1, d=0):
        super(LearnedAutoencoderWithVarIQImbalance, self).__init__()
        self.encoder = ComplexUnitModulus(input_dim,encoding_dim)
        self.encoding_dim = encoding_dim
        self.variance = variance
        self.b = b
        self.d = d
        layers = []
        prev_dim = encoding_dim*2
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim,dim*2))
            layers.append(nn.ReLU())
            prev_dim = dim*2
        self.decoder = nn.Sequential(
            *layers,
            nn.Linear(prev_dim,input_dim*2)
        )

    def forward(self,x):
        encoder_out = self.encoder(x)
        self.r = torch.tensor(0.5*(1+self.b*np.exp(1j*self.d)), dtype=torch.complex64)
        y_real = encoder_out[:, :self.encoding_dim]
        y_imag = encoder_out[:, self.encoding_dim:]
        y = torch.complex(y_real,y_imag)
        yiq = self.r * y + (1-self.r.conj()) * (y.conj())
        yiqr = yiq.real
        yiqi = yiq.imag
        yiqstack = torch.cat((yiqr,yiqi),dim=1)
        noise_np = np.random.normal(0,self.variance,size=self.encoding_dim*2)
        noise_tensor = torch.tensor(noise_np,dtype=torch.float)
        y_iq_stack_noisy = yiqstack + noise_tensor
        return self.decoder(y_iq_stack_noisy)

    class AdversarialNetwork(nn.Module):
        def __init__(self, input_dim, encoding_dim, hidden_dec_dims, hidden_pred_dims, variance):
            self.encoder = ComplexUnitModulus(input_dim, encoding_dim)
            self.encoding_dim = encoding_dim
            self.variance = variance
            dec_layers = []
            prev_dim = encoding_dim * 2
            for dim in hidden_dec_dims:
                dec_layers.append(nn.Linear(prev_dim, dim * 2))
                dec_layers.append(nn.ReLU())
                prev_dim = dim * 2
            self.decoder = nn.Sequential(
                *dec_layers,
                nn.Linear(prev_dim, input_dim * 2)
            )
            pred_layers = []
            prev_dim = encoding_dim * 2
            for dim in hidden_pred_dims:
                pred_layers.append(nn.Linear(prev_dim, dim * 2))
                pred_layers.append(nn.ReLU())
                prev_dim = dim * 2
            self.predictor = nn.Sequential(
                *pred_layers,
                nn.Linear(prev_dim, 2),
            )

        def forward(self, x):
            # Encoder output
            encoder_out = self.encoder(x)

            # Predictor output, we scale the output by our predefined ranges (softmax normalizes between [0,1]) and add bias
            predictor_out = self.predictor(encoder_out)  # Size: B x 2 where B is batch size
            b_raw = predictor_out[:, 0]
            d_raw = predictor_out[:, 1]
            b_tensor = torch.tanh(d_raw) * 0.1 + 0.9  # Scale between 0.1 and -0.1 and add bias
            d_tensor = torch.tanh(
                b_raw) * np.pi / 16 + np.pi / 16  # We scale it between pi/16 and -pi/16 and then add pi/16

            cos_d_tensor = torch.cos(d_tensor)
            sin_d_tensor = torch.sin(d_tensor)
            K1R_tensor = 0.5 * (1 + b_tensor * cos_d_tensor)
            K1I_tensor = 0.5 * b_tensor * sin_d_tensor
            K2R_tensor = 0.5 * (1 - b_tensor * cos_d_tensor)
            K2I_tensor = 0.5 * b_tensor * sin_d_tensor

            # Some nice information about the mean b and d distortion
            self.b_mean = torch.mean(b_tensor)
            self.d_mean = torch.mean(d_tensor)

            # Here we add IQ imbalance
            y_real = encoder_out[:, :self.encoding_dim]  # Size: B x E where E is encoding dimension
            y_imag = encoder_out[:, self.encoding_dim:]  # Size: B x E where E is encoding dimension
            y_IQ_real = (K1R_tensor + K2R_tensor).unsqueeze(1) * y_real + (-K1I_tensor + K2I_tensor).unsqueeze(
                1) * y_imag
            y_IQ_imag = (K1R_tensor + K2I_tensor).unsqueeze(1) * y_real + (K1R_tensor - K2R_tensor).unsqueeze(
                1) * y_imag
            y_IQ = torch.cat([y_IQ_real, y_IQ_imag], dim=1)

            # Finally we add noise
            noise = torch.randn_like(y_IQ) * self.variance
            y_IQ_noisy = y_IQ + noise

            # And run the decoder
            return self.decoder(y_IQ_noisy)


"""
OMP/PSOMP
"""

def omp(A, y, epsilon, max_iterations=np.inf):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse signal recovery.
    
    Parameters:
    D : np.ndarray
        The sensing matrix (size: m x n).
    y : np.ndarray
        The observed vector (size: m x 1).
    epsilon: float
        The error tolerance for stopping criterion.
    max_iterations : int
        Maximum number of iterations to perform.
        
    Returns:
    x_hat : np.ndarray
        The recovered sparse signal (size: n x 1).
    """
    m, n = A.shape
    x_hat = np.zeros((n, 1))
    residual = y.copy()
    index_set = []
    max_iterations = min(max_iterations, n)
    err = np.inf
    while err > epsilon and len(index_set) < max_iterations:
        # Step 1: Find the index of the atom that best correlates with the residual
        correlations = A.T @ residual
        correlations[index_set] = 0
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)

        # Step 2: Solve the least squares problem to update the coefficients
        A_subset = A[:, index_set]
        x_subset, _, _, _ = np.linalg.lstsq(A_subset, y, rcond=None)

        # Step 3: Update the residual
        residual = y - A_subset @ x_subset
        err = np.linalg.norm(residual)

    # Step 4: Construct the full solution vector
    for i, idx in enumerate(index_set):
        x_hat[idx] = x_subset[i]

    return x_hat


def psomp(A, y, K, sigma2=None):
    """
    Paired-Support Orthogonal Matching Pursuit (PSOMP)
    Based on Algorithm 1 in Masoumi & Myers (2023).

    Inputs:
        A      : sensing matrix (M x N)
        y      : measurement vector (M,)
        K      : sparsity level of x
        sigma2 : noise variance (optional for stopping rule)

    Outputs:
        z_hat  : estimated augmented sparse vector (2N,)
    """

    M, N = A.shape

    # Build augmented matrix
    A_aug = np.hstack([A, np.conjugate(A)])

    # Initialize
    r = y.copy()
    Q = []     # support set
    z_hat = np.zeros(2*N, dtype=complex)

    max_iter = 2*K
    err = np.inf
    while len(Q) < max_iter and (sigma2 is None or err > sigma2):

        # --- Step 1: support detection (paired) ---
        # Compute both matching terms
        match1 = np.abs(np.conjugate(A).T @ r)     # |a_j^* r|
        match2 = np.abs(A.T @ r)                   # |a_j^T r|

        # Choose best index from first N entries
        j = np.argmax(np.maximum(match1[:N], match2[:N]))

        # Paired support structure
        pair = [j, j+N]
        Q.extend(pair)

        # --- Step 2: least squares on selected support ---
        A_sub = A_aug[:, Q]
        z_sub, *_ = np.linalg.lstsq(A_sub, y, rcond=None)

        # assign
        for ii, idx in enumerate(Q):
            z_hat[idx] = z_sub[ii]

        # --- Step 3: update residual ---
        r = y - A_sub @ z_sub

    return z_hat

def find_x_xi(z : np.ndarray):
    """
    Function to recover the original signal and IQ imbalance parameter from the IQ imbalanced signal.
    Parameters:
        z : np.ndarray
            IQ imbalanced signal (size: 2n x 1).
            
        Returns:
        x : np.ndarray
            Recovered original signal (size: n x 1).
        xi : float
            Estimated IQ imbalance parameter.
    """
    z_1,z_2 = np.split(z, 2)
    alpha = np.linalg.norm(z_1)**2
    beta = np.linalg.norm(z_2)**2
    gamma = z_1.T @ z_2
    xi_hat = (alpha - beta - 2*gamma + np.sqrt( (alpha - beta)**2 + 4*np.abs(gamma)**2))/(2*(alpha - beta + np.conj(gamma) - gamma))
    x_hat = (np.conjugate(xi_hat)*z_1 + (1-xi_hat)*np.conjugate(z_2)) / (abs(xi_hat)**2 + abs(1-xi_hat)**2)
    return x_hat.reshape(-1,1), xi_hat