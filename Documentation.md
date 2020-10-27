# TinyFlow - A very (very) Simpy ANN Framework
# Documentation
## **!! This document is under construction !!**

TinyFlow was built as a fun side project to allow people to build and play around with simple Neural Networks, without having to go through complex installation processes - while also presenting them an opportunity to contribute to the framework itself.

The aim of this project is to allow people to not only build these neural networks, but also look behind the scenes into the source code of the framework to understand how every single component works. This is the reason every single component is documented (as much as possible, will be adding more comprehensive explanations soon).

One of the most exciting features of the framework is a wrapper component, called Network. It allows you to simplify and shorten your code to just a couple of lines of code allowing you to focus more on prototyping and less on writing code.

This document simply takes you through on how to get started with this framework and serves as a reference to all the different components the Framework makes available to you.

# **Quick Start**
The fastest way to get started with building neural networks in TinyFlow is to take a look at the demo files available in the root of the repository (also linked in the readme) and take a look at the code. TinyFlow was built with an aim to make it easy to learn, read, and implement neural networks.
The demo files implement a few common algorithms/networks which you might want to implement, and can serve as a good starting point to move forward.
The demos walk you through the following:
- Linear Regression
- Linear Regression using Network
- Binary Logistic Regression
- Binary Logistic Regression using Network
- Categorical Classification
- Categorical Classification using Network

# **Index**
### **The Bare-Bones Framework**

- [Generating random data](#generating-random-data)
- [Layers](#layers)
- [Activation Functions](#activation-functions)
- Optimizers
- Loss Function
- Modelling an actual network
- Setting up a training loop
- Testing your model 


### **The Network Wrapper**
- The Network Class
- Easily adding Layers and Activation Functions to your model
- Training your model
- Testing your model

---
# The Bare-Bones framework
## **Generating random data**
The framework allows you to generate random classification training data. The algorithm generates 2 dimensional points, divided into a *user-defined* number of classes.

The code for this has been taken from [CS231N's public website](https://cs231n.github.io/neural-networks-case-study/), thanks Stanford!

**Usage**
```
X, y = spiral_data(number_of_examples, number_of_classes)
```
Where X contains the training samples, and y contains the labels corresponding to each of the samples.

## **Layers**
The framework currently allows you to define 2 types of layers:
- **Dense / Fully Connected**
  - **Usage**
  ```
  dense1 = Layer_Dense(number_of_inputs, number_of_neurons, weight_regularizer_l1, weight_regularizer_l2, bias_regulariser_l2, bias_regulariser_l2)
  ```
  - This instantiates a fully connected layer, with user defined number of inputs and neurons. You have optional arguments available to define the regularization strength for L1 and L2 regularization.
- **Dropout**
  - **Usage**
  ```
    dropout1 = Layer_Dropout(dropout_rate)
  ```
  - Instantiates a Dropout layer, where dropout_rate defines the fraction of neurons that will be dropped.

## **Activation Functions**
The framework includes 4 main kinds of activation functions:
- **ReLU**
  - **Usage** (Checkout demo files for elaborate usage instructions)

  ```
  ...
  
  activation1 = Activation_ReLU()
  
  ...
  
  // inside the training loop

  activation1.forward(dense1.output)
  ```

  ```
  ...
  
  // Using ReLU with Network

  model = Network()

  ...

  model.add(Activation_ReLU())
  
  ...
  
  ```

- **Softmax**
  - **Usage** (Checkout demo files for elaborate usage instructions)

   ```
  ...
  
  activation2 = Activation_Softmax()

  
  ...
  
  // inside the training loop

  activation2.forward(dense2.output)

  ```

  ```
  ...
  
  // Using ReLU with Network

  model = Network()

  ...

  model.add(Activation_Softmax())
  
  ...
  
  ```

  - **Sigmoid**
  - **Usage** (Checkout demo files for elaborate usage instructions)

   ```
  ...
  
  activation3 = Activation_Sigmoid()

  
  ...
  
  // inside the training loop

  activation3.forward(dense2.output)

  ```

  ```
  ...
  
  // Using ReLU with Network

  model = Network()

  ...

  model.add(Activation_Sigmoid())
  
  ...
  
  ```