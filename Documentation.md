# TinyFlow - A very (very) Simpy ANN Framework
# Documentation
## **!! This document is under construction !!**

TinyFlow was built as a fun side project to allow people to build and play around with simple Neural Networks, without having to go through complex installation processes - while also presenting them an opportunity to contribute to the framework itself.

The aim of this project is to allow people to not only build these neural networks, but also look behind the scenes into the source code of the framework to understand how every single component works. This is the reason every single component is documented (as much as possible, will be adding more comprehensive explanations soon).

One of the most exciting features of the framework is a wrapper component, called Network. It allows you to simplify and shorten your code to just a couple of lines of code allowing you to focus more on prototyping and less on writing code.

This document simply takes you through on how to get started with this framework and serves as a reference to all the different components the Framework makes available to you.

# **Index**
### **The Bare-Bones Framework**

- [Generating random data](#generating-random-data)
- [Layers](#layers)
- Activation Functions
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
  fc1 = dense1 = Layer_Dense(number_of_inputs, number_of_neurons, weight_regularizer_l1, weight_regularizer_l2, bias_regulariser_l2, bias_regulariser_l2)
  ```
  - This instantiates a fully connected layer, with user defined number of inputs and neurons. You have optional arguments available to define the regularization strength for L1 and L2 regularization.
- **Dropout**
  - **Usage**
  ```
    dropout1 = Layer_Dropout(dropout_rate)
  ```
  - Instantiates a Dropout layer, where dropout_rate defines the fraction of neurons that will be dropped.