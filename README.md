# DeepNetToolKit

Elements and modules needed for training and using "Deep Neural Networks" for experimental tasks.

Current Version covers following:
- Common types of activation function (ReLU, Sigmoid, tanh, softmax, etc.)
- Common types of layer architectures (fully-connected, convolutional, pooling, weight sharing, etc.)
- Common types of error functions (Zero-One, Euclidean distance, Log-Likelihood, Cross-Entropy, etc.)
- Common types of cost functions (L1-norm, L2-norm, contractive, etc.)
- Common types of weight update procedures (direct, momentom, nesterov momentum, etc.)
- Common types of pre-training (Auto-Encoders, RBMs, Convolutional-Auto-Encoders, etc.)
- Common types of networks (Staked Auto-Encoders, Staked RBMs, Convolutional Nets, Belief Nets, etc.)
- Common types of learning procedures (early stopping, etc.)
- Tools for training, testing, storing, and using Neural Nets.

These modules use Theano Library (a mathematical library specially designed for Deep Learning projects)
as their underlying library. Theano convert your code to C++ code and compile it in real time,
so you should not worry about performance that much. It also has the ability to run your code on GPU 
if you have a CUDA compatible GPU on your computer and nvcc installed.

Some parts of codes are based on Theano deep learning tutorial.

# Installation
TODO

# Quick Start
TODO
