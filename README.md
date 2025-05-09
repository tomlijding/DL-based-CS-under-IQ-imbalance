# Deep Learning-Based CS under IQ imbalance For Channel Estimation of mmWave phased arrays

## Introduction
In this repository, we approach the problem of IQ imbalance in channel estimation of mmWave phased arrays from a compressed sensing (CS)/neural network point of view.

We refer the interested reader to the pdf file Data_Compression__Deep_Learning_Based_CS_under_IQ_Imbalance.pdf for a full report on the problem and the approach.

We are able to find a remarkably small neural network solution, able to completely negate the effects of IQ imbalance, as well as faithfully reconstruct the signal for a range of SNR. Plots of NMSE for various models are shown below

![allmodels](https://github.com/user-attachments/assets/f38a920e-4939-41fa-851e-0a847b25c71f)

Additionally, we are able to restrict the values of the CS matrix even further, to a discrete set defined by $q \in \{ \pm \pi, \pm \frac{1}{2}\pi, 0\}$ corresponding to a 5-bit limited resolution phased-array antenna, allowing the deployment on low-cost and complexity phased-array antenna setup. The results are shown below.

![discrete_model_performance_page-0001](https://github.com/user-attachments/assets/09642c33-319e-4c33-bbe3-0b4de7f5ab7e)

Below we show how a signal is reconstructed using the algorithm

![reconstruction_noiseless](https://github.com/user-attachments/assets/3bd6b4a3-eb62-430d-b83b-3d70bf06bc5d)


## Methodology
We use autoencoder based compressed sensing to reconstruct the chanel from limited measurements, allowing us to find a good approximation of the original channel of size 100 from just 40 measurements.

A visual representation of the measurement model is given below

![channel_image (1)_page-0001](https://github.com/user-attachments/assets/30c7f702-63ce-45fc-a284-52543d634518)

where we model a phased-array antenna setup via a compressed sensing matrix. A known symbol is sent through the channel, resulting in the vector $h$, which is measured at $N$ timesteps where at each timestep the 
phased array setup is adjusted. The phased-array setup is then modelled as a restricted CS matrix, where all values are restricted to $e^{jq}$. 

This matrix is trained via the Pytorch architecture via an autoencoder setup shown below

![autoencoder](https://github.com/user-attachments/assets/b73ca748-e19a-4edb-9a44-7ad1084423b1)

We then add noise, reduce the amount of measurements and add IQ imbalance. The IQ imbalance is a result of an imbalance in the phase and amplitude of the local oscillator (LO) at the receiver (RX) end. A schematic is given below, courtesy of [^1]

![IQI-Model](https://github.com/user-attachments/assets/f0144932-f688-478a-ab06-8b3ded75671b)


[^1]: A. -A. A. Boulogeorgos, V. M. Kapinas, R. Schober and G. K. Karagiannidis, "I/Q-Imbalance Self-Interference Coordination," in IEEE Transactions on Wireless Communications, vol. 15, no. 6, pp. 4157-4170, June 2016, doi: 10.1109/TWC.2016.2535441.

