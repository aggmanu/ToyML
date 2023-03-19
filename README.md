# ToyML

Broad aim: Play with Python and Machine learning (ML).

More specific aim: Analyze noisy high-resolution discrete time-series from $N$ sources, say $s_i(m \Delta t)\, 1 \leq i \leq N\, 0 \leq m \Delta t \leq T.$ Examples, stock market, brain resting-state MEG, heart ECG etc. 'Analyze' here means, given an input of time-window length $T$ from all sources, can we predict anything in the future?


Heart ECG: Has clear patterns. ML has been used to predict heart conditions.
Stock market: What is this garbage?
Resting state MEG: Hopefully some meaningful patterns.

What might we mean 'predicting the future'? To begin with, it can be whether $s_i(T+k)/s_i(0) > f,$ where $f > 1$ is fixed parameter. We do not know what values $k$ should be. It can be interesting to find out $k$ for which predictions can be made with higher probability.

ML/Neural Network (NL) model names that I have heard of: basic multiple layers, convolutional NN (CNN), variational autoencoder (VAE), generative adversarial networks (GAN), neural differential equations. Each model can have different set of hyperparameters. For example, can be choice of activation function, learning rate, number of input nodes, number of output nodes, number of hidden layers etc.

Proposed Python module to use for ML model: pyTorch.

Proposed steps:
1. Get stock market data and save as csv file or numpy or pandas -- https://towardsdatascience.com/python-how-to-get-live-market-data-less-than-0-1-second-lag-c85ee280ed93. (easy, but there were some issues with this module in 2021 which, I think, were later resolved in their github)
2. Discuss input and output for ML model. (very difficult)
3. Implement basic multiple layers using pyTorch. (easy?)
4. Training and validation scheme. (difficult)
5. Celebrate when we see any output on testing data even though it might be garbage non-prediction --> Refine model by changing input data, traininig criteria, loss function, hyper parameters, different ML model etc.

# Some terminology to be on the same page:

Bold-face are *tensors*. In ML terminology, tensor just means high-dimensional matrices.

*Input* tensor to a model is $\mathbf{x}.$ Input can be $n$ observations of $k$ *features*.

*True output* tensor is $\mathbf{y}.$

*Parameter* tensor of a model is $\mathbf{\theta}.$

*Model* $f$ is a function of unknown parameters and inputs. Various architectures -- egs. CNNs, VAEs, GANs, RNN etc.

*Model output* tensor of a model is $\mathbf{\widehat{y}} = f_\theta (\mathbf{x}).$

Loss function is $l(\mathbf{y}, \mathbf{\widehat{y}}).$

ML model trains on a *batch* of input tensors to estimate parameters with objective of minimizing loss function.

*Hyperparameters* are thresholds and properties of the model, for a given model architecture.



