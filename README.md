![Cover](https://i.imgur.com/0YF4BGU.png)
# TinderFlow - A very (very) simple ANN Framework
### I built this framework as a small side project for educational purposes to allow people new to the field of Deep Learning to try their hands at building simple Neural Networks with minimal setup. The only dependency for the framework is NumPy.

The framework while being extremely easy to use also allows a relatively high degree of customizability, from being able to configure every single Optimizer's hyperparameters to generating your own training data right inside the framework allows for an indefinite amount of possibilities.

The [demo.py](demo.py) file contains example code to get you started.

## *Features*
- Fully connected Layers
- ReLU, Softmax activations
- Generate training data (Classification data)
- Categorical Cross Entropy Loss
- Backpropogation
- Optimizers
  - SGD (with decay and momentum)
  - AdaGrad
  - RMSprop
  - Adam

---

## *Under Development*
Files with a double underscore prefix are under development
- Wrapper to allow easier API integration/development
  - The wrapper would allow users to create NNs without having to worry about the dimensions for each layer, and just focus on experimentation.

---
The name is just a joke that I made by combining the second half of my name (Jyotinder) with TensorFlow.