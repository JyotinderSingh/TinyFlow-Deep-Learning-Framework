![Cover](https://i.imgur.com/2Xp9nxM.png)
# TinyFlow - A very (very) simple ANN Framework
### I built this framework as a small side project for educational purposes to allow people new to the field of Deep Learning to try their hands at building simple Neural Networks with minimal setup.
### The only dependency for the framework is [NumPy](https://numpy.org/)!

The framework while being extremely easy to use also allows a relatively high degree of customizability, from being able to configure every single Optimizer to generating your own classification training data right inside the framework allows for a ton of possibilities.

All implementations are fully documented to help understand the underlying logic for each component better.


#### The following files contain example code for using the Network wrapper.
- [demo_Network_categorical.py](./demo_Network_categorical.py)
- [demo_Network_binary_logistic_regression.py](./demo_Network_binary_logistic_regression.py)
- [demo_Network_linear_regression.py](./demo_Network_linear_regression.py)

#### The following files contain example code to get you started.
- [demo_categorical.py](./demo_categorical.py) 
- [demo_binary_logistic_regression.py](./demo_binary_logistic_regression.py)
- [demo_linear_regression.py](./demo_linear_regression.py)

## **Features**
- Fully connected, Dropout Layers
- ReLU, Softmax, Sigmoid, and Linear activations
- Generate synthetic training data (Classification, and Regression data)
- Categorical Cross Entropy Loss, Binary Cross Entropy Loss, Mean Squared Error
- L1 & L2 regularization
- Backpropogation
- Optimizers
  - SGD (with decay and momentum)
  - AdaGrad
  - RMSprop
  - Adam
- Network Wrapper with TinyFlow backend
  - The wrapper supports setting up a Neural Network and running a simple training loop using the above components, without having to write long error prone code.

## Steps to use
```
- Make sure you have NumPy installed.
- Clone the repository to your system and place the TinyFlow folder in your project directory.
- Follow the steps in the demo.py to build your own network. 
```


---

## *Under Development*
- Documentation

### **Contributions and Bug Fixes are welcome!**

---