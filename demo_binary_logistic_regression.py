# Binary Logistic Regression Demo
import numpy as np
from TinyFlow.Model import spiral_data, model_accuracy_sigmoid
from TinyFlow.Layers import Layer_Dense
from TinyFlow.Activations import Activation_ReLU, Activation_Sigmoid
from TinyFlow.Loss import Loss_BinaryCrossEntropy
from TinyFlow.Optimizers import Optimizer_Adam

# Create dataset
X, y = spiral_data(100, 2)

# Reshape the labels as they aren't sparse anymore, They're binary, 0 & 1
# Reshape the labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
#
# We do this reshaping as spiral data values are mapped directly to
# the sparse class values taht were the ideal "one hot index from
# the network's output". However in this case we're trying to represent
# Binary output. IN this example we have a single output neuron, of a target
# value of either 0 or 1.
y = y.reshape(-1, 1)

# Create a dense layer with 2 input features and 3 output values
# first dense layer, 2 inputs (each sample has 2 features), 64 outputs
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regulariser_l2=5e-4)

# Create ReLU activation
activation1 = Activation_ReLU()

# Create second dense layer with 64 input features
# (as we take output of previous layer here) and 1 output
dense2 = Layer_Dense(64, 1)

# Create Sigmoid Activation
activation2 = Activation_Sigmoid()

# Create a loss function
loss_function = Loss_BinaryCrossEntropy()

# Create an optimizer
optimizer = Optimizer_Adam(decay=1e-8)

# Train in loop
for epoch in range(10001):

    # Make a forward pass of our training adaa through this layer
    dense1.forward(X)

    # Make a forward pass through our activation function
    activation1.forward(dense1.output)

    # Make forward pass through second dense layer
    dense2.forward(activation1.output)

    # Make a forward pass through the second activation function
    activation2.forward(dense2.output)

    # Calculate the losses from the second activation function
    sample_losses = loss_function.forward(activation2.output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)

    # Overall loss
    loss = data_loss + regularization_loss

    # Calculate accuracy from output of activation2 and targets
    # Part in the brackets returns a binary maskk - array consisting
    # of True/False values, multiplying it by 1 changes it into array of 1s and 0s
    accuracy = model_accuracy_sigmoid(activation2.output, y)

    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate:.3f}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activation1.backward(dense2.dvalues)
    dense1.backward(activation1.dvalues)

    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate model

# Create test dataset
X_test ,y_test = spiral_data(100, 2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

# Make a forward pass of the testing data though this layer
dense1.forward(X_test)

# Make a forward pass through the activation function
activation1.forward(dense1.output)

# Make a forward pass through the second dense layer
dense2.forward(activation1.output)

# Make a forward pass through the second activation function
activation2.forward(dense2.output)

# Calculate the sample loses from output of activation2 and targets
sample_losses = loss_function.forward(activation2.output, y_test)

# Calculate mean loss
loss = np.mean(sample_losses)

# Calculate accuracy over test data
accuracy = model_accuracy_sigmoid(activation2.output, y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')