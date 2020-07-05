# Linear Regression Demo
import numpy as np
from TinyFlow.Datasets import sine_data
from TinyFlow.Layers import Layer_Dense
from TinyFlow.Activations import Activation_ReLU, Activation_Linear
from TinyFlow.Metrics import model_accuracy_linear
from TinyFlow.Loss import Loss_MeanSquaredError
from TinyFlow.Optimizers import Optimizer_Adam

# Create dataset
X, y = sine_data()

# Create dense layer with 1 input feature and 64 outputs
# The one input feature is the x coordinate of the sin wave
dense1 = Layer_Dense(1, 64)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second dense layer, with 64 inputs (output dims of prev layer)
# and 64 outputs
dense2 = Layer_Dense(64, 64)

# Create ReLU activation
activation2 = Activation_ReLU()

# Create final output layer, with 1 output for regressed result
dense3 = Layer_Dense(64, 1)

# Linear Activation for the final layer
activation3 = Activation_Linear()

# Create Loss function
loss_function = Loss_MeanSquaredError()

# Create optimizer
optimizer = Optimizer_Adam(decay=1e-8)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
# 500 defines the strictness of loss calculation
# Use larger number for more strictness
accuracy_precision = np.std(y) / 500

# Train in loop
for epoch in range(10001):

    # Make a forward pass of our training data through the first layer
    dense1.forward(X)

    # Make a forward pass through the activation function
    activation1.forward(dense1.output)

    # Make a forward pass through the second dense layer
    dense2.forward(activation1.output)

    # Make a forward pass through the activation function
    activation2.forward(dense2.output)

    # Make a forward pass through the third dense layer
    dense3.forward(activation2.output)

    # Make a forward pass through the activation function
    activation3.forward(dense3.output)

    # Calculate sample losses from output of activation3
    sample_losses = loss_function.forward(activation3.output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)

    # Calculate overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and target values
    accuracy = model_accuracy_linear(activation3.output, y, accuracy_precision)

    if not epoch % 100:
        print(
            f'\nepoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, (data loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate:.5f}')

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dvalues)
    dense3.backward(activation3.dvalues)
    activation2.backward(dense3.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()