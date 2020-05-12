![Cover](https://i.imgur.com/0YF4BGU.png)
# TinderFlow - A very (very) simple ANN Framework
### I built this framework as a small side project for educational purposes to allow people new to the field of Deep Learning to try their hands at building simple Neural Networks with minimal setup. The only dependency for the framework is NumPy.

The framework while being extremely easy to use also allows a relatively high degree of customizability, from being able to configure every single Optimizer to generating your own classification training data right inside the framework allows for a ton of possibilities.
All implementations are fully documented to help understand the underlying logic for each component better.

The [demo.py](demo.py) file contains example code to get you started.

The [__NetworkDemo.py](__NetworkDemo.py) file contains example code for using the Network wrapper.

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
- Network Wrapper with TinderFlow backend (under development)
  - The wrapper currently supports setting up a basic Neural Network and running a simple training loop using the above components, without having to worry about instantiating and getting the dimensions right for every layer.

---

## *Under Development*
Files with a double underscore prefix are under development
- Wrapper to allow easier API integration/development
  - Testing pending

### **Contributions and Bug Fixes are welcome!**

---
The name for the framework is just a joke that I made by combining the second half of my name (Jyotinder) with TensorFlow.